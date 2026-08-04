"""Claim-bearing Prob4D factors with native block-sparse gauge inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._gauge_aware_contracts import (
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareObservationBatch,
)
from .explicit_gauge_prob4d import (
    PROB4D_FACTOR_API_VERSION,
    PROB4D_FACTOR_BUNDLE_SCHEMA_VERSION,
    _array_sha256,
    _assert_stack_matches_bundle,
    _calibration_ids,
    _expected_sparse_stack,
    _require_integer,
    _require_sha256,
    _require_string,
    _sparse_stack_sha256,
    _SparseStackProtocol,
    _string_tuple,
    _validate_envelope_and_bundle,
    _validate_linearization,
    _validate_stack,
    _ValidatedBundleProtocol,
)
from .physical_linearization import PhysicalLinearizationV1
from .prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from .prior_aware_sparse_gauge_belief import (
    SparseGaugeAwareObservationBatch,
    update_prior_aware_sparse_gauge_belief,
)
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1

NATIVE_SPARSE_EXPLICIT_GAUGE_SOLVER_VERSION = 1


@dataclass(frozen=True, slots=True)
class SparseExplicitGaugeFactorAdapterResult:
    """A strict factor envelope prepared without dense gauge expansion."""

    batch: SparseGaugeAwareObservationBatch
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    dense_equivalent_gauge_design_bytes: int
    gauge_ids: tuple[str, ...]
    view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.batch, SparseGaugeAwareObservationBatch):
            raise TypeError("batch must be a SparseGaugeAwareObservationBatch")
        object.__setattr__(
            self,
            "observation_artifact_id",
            _require_sha256(
                self.observation_artifact_id,
                name="observation_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "linearization_artifact_id",
            _require_sha256(
                self.linearization_artifact_id,
                name="linearization_artifact_id",
            ),
        )
        object.__setattr__(
            self,
            "provider_manifest_id",
            _require_sha256(
                self.provider_manifest_id,
                name="provider_manifest_id",
            ),
        )
        object.__setattr__(
            self,
            "calibration_artifact_ids",
            _calibration_ids(self.calibration_artifact_ids),
        )
        object.__setattr__(
            self,
            "runtime_revision_source",
            _require_string(
                self.runtime_revision_source,
                name="runtime_revision_source",
            ),
        )
        object.__setattr__(
            self,
            "dense_equivalent_gauge_design_bytes",
            _require_integer(
                self.dense_equivalent_gauge_design_bytes,
                name="dense_equivalent_gauge_design_bytes",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "gauge_ids",
            _string_tuple(self.gauge_ids, name="gauge_ids"),
        )
        object.__setattr__(
            self,
            "view_ids",
            _string_tuple(self.view_ids, name="view_ids"),
        )


def build_claim_bearing_sparse_explicit_gauge_batch(
    validated_bundle: _ValidatedBundleProtocol,
    sparse_stack: _SparseStackProtocol,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    anchor_correlation_group_ids: tuple[str, ...] | None = None,
    anchor_prior_reliability: np.ndarray | None = None,
    anchor_prior_nominal_probability: np.ndarray | None = None,
    anchor_composite_weight: np.ndarray | None = None,
    anchor_bias_jacobian: np.ndarray | None = None,
    anchor_bias_prior_covariance: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SparseExplicitGaugeFactorAdapterResult:
    """Validate claim-bearing factors and retain their local gauge blocks."""

    (
        artifact_id,
        provider_manifest_id,
        calibration_ids,
        runtime_source,
        gauge_ids,
        observation_count,
    ) = _validate_envelope_and_bundle(validated_bundle)
    causal_frame_stop = _require_integer(
        validated_bundle.envelope.causal_frame_stop,
        name="causal_frame_stop",
        minimum=1,
    )
    expected_stack = _expected_sparse_stack(
        validated_bundle.bundle,
        gauge_ids=gauge_ids,
    )
    if expected_stack["bundle_observation_count"] != observation_count:
        raise ValueError("factor bundle differs from envelope observation_count")
    selected_observation_count = int(expected_stack["selected_observation_count"])
    stack = _validate_stack(
        sparse_stack,
        gauge_ids=gauge_ids,
        causal_frame_stop=causal_frame_stop,
        selected_observation_count=selected_observation_count,
    )
    _assert_stack_matches_bundle(stack, expected_stack)
    _validate_linearization(
        linearization,
        observation_artifact_id=artifact_id,
        frame_indices=stack["frame_indices"],
        point_ids=stack["point_ids"],
        view_ids=stack["view_ids"],
        gauge_indices=stack["gauge_indices"],
    )

    physical_prediction = np.asarray(physical_prediction_xyz_m, dtype=np.float64)
    if physical_prediction.shape != (selected_observation_count, 3):
        raise ValueError("physical_prediction_xyz_m must have shape (M, 3)")
    if not np.all(np.isfinite(physical_prediction)):
        raise ValueError("physical_prediction_xyz_m must be finite")
    shared = (
        np.zeros((selected_observation_count, 3, 0), dtype=np.float64)
        if shared_bias_jacobian is None
        else np.asarray(shared_bias_jacobian, dtype=np.float64)
    )
    view = (
        np.zeros((selected_observation_count, 3, 0), dtype=np.float64)
        if view_bias_jacobian is None
        else np.asarray(view_bias_jacobian, dtype=np.float64)
    )
    row_power = stack["association"] * stack["composite"]
    if np.any(row_power <= 0.0):
        raise ValueError("association-weighted composite power must be positive")

    gauge_dimension = int(stack["gauge_prior"].shape[0])
    dense_equivalent_bytes = (
        selected_observation_count * 3 * gauge_dimension * np.dtype(np.float64).itemsize
    )
    extra_metadata = frozen_finite_json_mapping(metadata)
    reserved_metadata: dict[str, Any] = {
        "observation_artifact_id": artifact_id,
        "linearization_artifact_id": linearization.artifact_id,
        "baseline_belief_id": linearization.baseline_belief_id,
        "action_prefix_id": linearization.action_prefix_id,
        "simulator_revision": linearization.simulator_revision,
        "row_alignment_verified": True,
        "prob4d_claim_bearing_provider_v2_validated": True,
        "prob4d_claim_bearing_factor_api_version": PROB4D_FACTOR_API_VERSION,
        "prob4d_claim_bearing_factor_bundle_schema_version": (
            PROB4D_FACTOR_BUNDLE_SCHEMA_VERSION
        ),
        "prob4d_claim_bearing_factor_bundle_envelope_artifact_id": artifact_id,
        "prob4d_claim_bearing_sparse_stack_sha256": _sparse_stack_sha256(stack),
        "prob4d_sparse_stack_rederived_from_validated_bundle": True,
        "prob4d_bundle_observation_count": observation_count,
        "prob4d_selected_observation_count": selected_observation_count,
        "prob4d_claim_bearing_provider_manifest_id": provider_manifest_id,
        "prob4d_claim_bearing_calibration_artifact_ids": dict(calibration_ids),
        "prob4d_claim_bearing_runtime_revision_source": runtime_source,
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
        "prob4d_explicit_gauge_covariance_semantics": (
            "conditional-point-plus-explicit-joint-gauge-prior-v1"
        ),
        "prob4d_marginal_point_covariance_consumed": False,
        "prob4d_association_probability_semantics": (
            "generalized-Bayes-row-power-not-source-reliability-v1"
        ),
        "prob4d_association_probability_sha256": _array_sha256(stack["association"]),
        "prob4d_source_reliability_sha256": _array_sha256(stack["reliability"]),
        "prob4d_prior_nominal_probability_sha256": _array_sha256(stack["nominal"]),
        "prob4d_provider_composite_weight_sha256": _array_sha256(stack["composite"]),
        "prob4d_dense_compatibility_bridge": False,
        "prob4d_native_sparse_gauge_solver": True,
        "prob4d_native_sparse_gauge_solver_version": (
            NATIVE_SPARSE_EXPLICIT_GAUGE_SOLVER_VERSION
        ),
        "prob4d_dense_gauge_design_allocated": False,
        "prob4d_dense_equivalent_gauge_design_bytes": dense_equivalent_bytes,
        "prob4d_gauge_ids": list(gauge_ids),
        "prob4d_view_ids_canonical_order": sorted(set(stack["view_ids"])),
        "prob4d_factor_ids_sha256": hashlib.sha256(
            json.dumps(
                list(stack["factor_ids"]),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "physical_response_scale_source": (
            "PhysicalLinearizationV1.physical_response_m"
        ),
    }
    collisions = set(extra_metadata) & set(reserved_metadata)
    if collisions:
        raise ValueError(
            f"metadata overrides reserved explicit-gauge fields: {sorted(collisions)}"
        )
    extra_plain = plain_json(extra_metadata)
    if not isinstance(extra_plain, dict):
        raise TypeError("validated metadata lost its mapping type")
    batch_metadata = {**extra_plain, **reserved_metadata}

    base = GaugeAwareObservationBatch(
        innovation_m=stack["mean"] - physical_prediction,
        observation_covariance_m2=stack["conditional"],
        state_jacobian=linearization.state_jacobian,
        gauge_jacobian=np.zeros((selected_observation_count, 3, 0), dtype=np.float64),
        shared_bias_jacobian=shared,
        view_bias_jacobian=view,
        query_state_jacobian=linearization.query_state_jacobian,
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        correlation_group_ids=stack["groups"],
        prior_reliability=stack["reliability"],
        prior_nominal_probability=stack["nominal"],
        composite_weight=row_power,
        physical_response_scale_m=linearization.physical_response_scale_m,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=anchor_prior_nominal_probability,
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        metadata=batch_metadata,
        composite_weight_mode=COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    )
    batch = SparseGaugeAwareObservationBatch(
        base=base,
        local_gauge_jacobian=stack["local_gauge"],
        gauge_indices=stack["gauge_indices"],
        gauge_prior_covariance=stack["gauge_prior"],
    )
    return SparseExplicitGaugeFactorAdapterResult(
        batch=batch,
        observation_artifact_id=artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        provider_manifest_id=provider_manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_source,
        dense_equivalent_gauge_design_bytes=dense_equivalent_bytes,
        gauge_ids=gauge_ids,
        view_ids=stack["view_ids"],
    )


def update_claim_bearing_sparse_explicit_gauge_from_artifacts(
    validated_bundle: _ValidatedBundleProtocol,
    sparse_stack: _SparseStackProtocol,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    anchor_correlation_group_ids: tuple[str, ...] | None = None,
    anchor_prior_reliability: np.ndarray | None = None,
    anchor_prior_nominal_probability: np.ndarray | None = None,
    anchor_composite_weight: np.ndarray | None = None,
    anchor_bias_jacobian: np.ndarray | None = None,
    anchor_bias_prior_covariance: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ClaimBearingProb4DUpdateV1:
    """Run one strict native sparse explicit-gauge update or exact fallback."""

    adapted = build_claim_bearing_sparse_explicit_gauge_batch(
        validated_bundle,
        sparse_stack,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        anchor_correlation_group_ids=anchor_correlation_group_ids,
        anchor_prior_reliability=anchor_prior_reliability,
        anchor_prior_nominal_probability=anchor_prior_nominal_probability,
        anchor_composite_weight=anchor_composite_weight,
        anchor_bias_jacobian=anchor_bias_jacobian,
        anchor_bias_prior_covariance=anchor_bias_prior_covariance,
        metadata=metadata,
    )
    result = update_prior_aware_sparse_gauge_belief(
        adapted.batch,
        config=config,
    )
    return ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=adapted.linearization_artifact_id,
        provider_manifest_id=adapted.provider_manifest_id,
        calibration_artifact_ids=adapted.calibration_artifact_ids,
        runtime_revision_source=adapted.runtime_revision_source,
        runtime_revision_independently_verified=True,
    )


__all__ = [
    "NATIVE_SPARSE_EXPLICIT_GAUGE_SOLVER_VERSION",
    "SparseExplicitGaugeFactorAdapterResult",
    "build_claim_bearing_sparse_explicit_gauge_batch",
    "update_claim_bearing_sparse_explicit_gauge_from_artifacts",
]
