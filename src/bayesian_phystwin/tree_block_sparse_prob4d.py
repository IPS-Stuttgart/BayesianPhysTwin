"""Claim-bearing Prob4D updates using the block-tree Schur solver."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from .explicit_gauge_prob4d import (
    _calibration_ids,
    _require_sha256,
    _require_string,
)
from .physical_linearization import PhysicalLinearizationV1
from .prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from .prospective_prob4d_update import ClaimBearingProb4DUpdateV1
from .tree_block_sparse_gauge_belief import TreeBlockGaugeAwareBeliefResultV1
from .tree_block_sparse_gauge_belief_v2 import (
    update_tree_block_sparse_prior_aware_gauge_belief_v2 as update_tree_block_sparse_prior_aware_gauge_belief,
)
from .tree_sparse_explicit_gauge_prob4d import (
    build_claim_bearing_tree_sparse_prob4d_batch,
    load_claim_bearing_tree_sparse_prob4d,
)

CLAIM_BEARING_TREE_BLOCK_PROB4D_SCHEMA = (
    "bayesian_phystwin.claim_bearing_tree_block_prob4d_update"
)
CLAIM_BEARING_TREE_BLOCK_PROB4D_VERSION = 1
CLAIM_BEARING_TREE_BLOCK_PROB4D_IDENTITY_VERSION = 1


def _canonical_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ClaimBearingTreeBlockProb4DUpdateV1:
    """Provider-bound result whose posterior remains tree-factorized."""

    result: TreeBlockGaugeAwareBeliefResultV1
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    _admission_id: str = field(init=False, repr=False, compare=False)
    _update_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.result, TreeBlockGaugeAwareBeliefResultV1):
            raise TypeError("result must be a TreeBlockGaugeAwareBeliefResultV1")
        for name in (
            "observation_artifact_id",
            "linearization_artifact_id",
            "provider_manifest_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        calibration_ids = _calibration_ids(self.calibration_artifact_ids)
        runtime_source = _require_string(
            self.runtime_revision_source,
            name="runtime_revision_source",
        )
        if type(self.runtime_revision_independently_verified) is not bool:
            raise TypeError("runtime_revision_independently_verified must be a bool")
        if self.runtime_revision_independently_verified is not True:
            raise ValueError("runtime_revision_independently_verified must be True")

        lineage = self.result.input_lineage
        expected = {
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "prob4d_claim_bearing_provider_manifest_id": (self.provider_manifest_id),
            "prob4d_claim_bearing_runtime_revision_source": runtime_source,
        }
        for name, value in expected.items():
            if lineage.get(name) != value:
                raise ValueError(f"result lineage does not bind {name}")
        lineage_calibration = _calibration_ids(
            lineage.get("prob4d_claim_bearing_calibration_artifact_ids")
        )
        if dict(lineage_calibration) != dict(calibration_ids):
            raise ValueError("result lineage does not bind calibration_artifact_ids")
        if (
            lineage.get("prob4d_claim_bearing_runtime_revision_independently_verified")
            is not True
        ):
            raise ValueError(
                "result lineage lacks independently verified runtime evidence"
            )

        object.__setattr__(
            self,
            "calibration_artifact_ids",
            frozen_finite_json_mapping(
                dict(calibration_ids),
                name="calibration_artifact_ids",
            ),
        )
        object.__setattr__(self, "runtime_revision_source", runtime_source)
        admission_id = _canonical_id(self.admission_descriptor())
        update_id = _canonical_id(
            {
                **dict(self.admission_descriptor()),
                "identity_version": (CLAIM_BEARING_TREE_BLOCK_PROB4D_IDENTITY_VERSION),
                "admission_id": admission_id,
                "tree_block_result_id": self.result.result_id,
            }
        )
        object.__setattr__(self, "_admission_id", admission_id)
        object.__setattr__(self, "_update_id", update_id)

    @property
    def inference_admissible(self) -> bool:
        return self.result.inference_admissible

    @property
    def accepted(self) -> bool:
        return self.inference_admissible

    @property
    def admission_id(self) -> str:
        return self._admission_id

    @property
    def update_id(self) -> str:
        return self._update_id

    @property
    def tree_block_result_id(self) -> str:
        return self.result.result_id

    @property
    def dense_covariance_materialized(self) -> bool:
        return False

    def admission_descriptor(self) -> Mapping[str, Any]:
        return frozen_finite_json_mapping(
            {
                "schema": CLAIM_BEARING_TREE_BLOCK_PROB4D_SCHEMA,
                "schema_version": CLAIM_BEARING_TREE_BLOCK_PROB4D_VERSION,
                "observation_artifact_id": self.observation_artifact_id,
                "linearization_artifact_id": self.linearization_artifact_id,
                "provider_manifest_id": self.provider_manifest_id,
                "calibration_artifact_ids": dict(self.calibration_artifact_ids),
                "runtime_revision_source": self.runtime_revision_source,
                "runtime_revision_independently_verified": (
                    self.runtime_revision_independently_verified
                ),
                "inference_admissible": self.result.inference_admissible,
                "reason": self.result.reason,
                "covariance_representation": (self.result.covariance.representation),
                "dense_covariance_materialized": False,
            },
            name="claim-bearing tree-block admission descriptor",
        )

    def descriptor(self) -> Mapping[str, Any]:
        return frozen_finite_json_mapping(
            {
                **dict(self.admission_descriptor()),
                "identity_version": (CLAIM_BEARING_TREE_BLOCK_PROB4D_IDENTITY_VERSION),
                "admission_id": self.admission_id,
                "tree_block_result_id": self.tree_block_result_id,
                "update_id": self.update_id,
            },
            name="claim-bearing tree-block Prob4D update descriptor",
        )

    def to_legacy(
        self,
        *,
        maximum_covariance_bytes: int | None = None,
    ) -> ClaimBearingProb4DUpdateV1:
        """Explicitly materialize the historical dense result."""

        return ClaimBearingProb4DUpdateV1(
            result=self.result.to_legacy(
                maximum_covariance_bytes=maximum_covariance_bytes
            ),
            observation_artifact_id=self.observation_artifact_id,
            linearization_artifact_id=self.linearization_artifact_id,
            provider_manifest_id=self.provider_manifest_id,
            calibration_artifact_ids=self.calibration_artifact_ids,
            runtime_revision_source=self.runtime_revision_source,
            runtime_revision_independently_verified=(
                self.runtime_revision_independently_verified
            ),
        )


def update_claim_bearing_tree_block_prob4d_from_artifacts(
    validated_observation: Any,
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
) -> ClaimBearingTreeBlockProb4DUpdateV1:
    """Run the strict claim-bearing update without a dense gauge-sized matrix."""

    adapted = build_claim_bearing_tree_sparse_prob4d_batch(
        validated_observation,
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
    result = update_tree_block_sparse_prior_aware_gauge_belief(
        adapted.batch,
        adapted.tree_gauge_design,
        config=config,
    )
    return ClaimBearingTreeBlockProb4DUpdateV1(
        result=result,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=adapted.linearization_artifact_id,
        provider_manifest_id=adapted.provider_manifest_id,
        calibration_artifact_ids=adapted.calibration_artifact_ids,
        runtime_revision_source=adapted.runtime_revision_source,
        runtime_revision_independently_verified=True,
    )


def update_claim_bearing_tree_block_prob4d_from_path(
    envelope_path: str | Path,
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
) -> ClaimBearingTreeBlockProb4DUpdateV1:
    """Load through Prob4D and run the block-tree claim-bearing update."""

    validated = load_claim_bearing_tree_sparse_prob4d(envelope_path)
    return update_claim_bearing_tree_block_prob4d_from_artifacts(
        validated,
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
        config=config,
        metadata=metadata,
    )


__all__ = [
    "CLAIM_BEARING_TREE_BLOCK_PROB4D_IDENTITY_VERSION",
    "CLAIM_BEARING_TREE_BLOCK_PROB4D_SCHEMA",
    "CLAIM_BEARING_TREE_BLOCK_PROB4D_VERSION",
    "ClaimBearingTreeBlockProb4DUpdateV1",
    "update_claim_bearing_tree_block_prob4d_from_artifacts",
    "update_claim_bearing_tree_block_prob4d_from_path",
]
