"""Strict Prob4D admission, candidate inference, and provenance binding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from ._canonical_contracts import plain_json
from ._gauge_aware_contracts import GaugeAwareBeliefResult
from .claim_bearing_prob4d import (
    build_claim_bearing_gauge_aware_batch_from_artifacts,
)
from .observation_belief import ObservationBeliefV1
from .physical_linearization import PhysicalLinearizationV1
from .posterior_covariance_semantics import (
    PosteriorCovarianceSemanticsV1,
    exact_prior_fallback_covariance_semantics,
    working_irls_covariance_semantics,
)
from .prior_aware_gauge_belief import PriorAwareGaugeConfigV1
from .prior_aware_gauge_belief_v2 import (
    update_prior_aware_gauge_belief_v2 as update_prior_aware_gauge_belief,
)

CLAIM_BEARING_PROB4D_UPDATE_VERSION = 1
CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION = 2
CLAIM_BEARING_PROB4D_INFERENCE_RESULT_VERSION = 1
CLAIM_BEARING_PROB4D_CANDIDATE_VERSION = 1
CLAIM_BEARING_PROB4D_CANDIDATE_IDENTITY_VERSION = 1
CLAIM_BEARING_PROB4D_CANDIDATE_RESULT_VERSION = 1


def _validated_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_calibration_ids(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("claim-bearing calibration artifact IDs are missing")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str):
            raise TypeError("calibration artifact names must be strings")
        if not isinstance(digest, str):
            raise TypeError(f"calibration artifact {name!r} digest must be a string")
        if not name:
            raise ValueError("calibration artifact names must be nonempty")
        result[name] = _validated_sha256(
            digest,
            name=f"calibration artifact {name}",
        )
    return MappingProxyType(dict(sorted(result.items())))


def _validated_runtime_revision_source(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("runtime_revision_source must be a string")
    if not value:
        raise ValueError("runtime_revision_source must be nonempty")
    return value


def _canonical_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype("<f8")))
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _inference_result_payload(result: GaugeAwareBeliefResult) -> dict[str, object]:
    if type(result.inference_admissible) is not bool:
        raise TypeError("result.inference_admissible must be a bool")
    if type(result.reason) is not str or not result.reason:
        raise ValueError("result.reason must be a nonempty string")
    arrays = {
        name: _array_descriptor(getattr(result, name))
        for name in (
            "state_coefficients",
            "gauge_delta",
            "shared_bias_coefficients",
            "view_bias_coefficients",
            "anchor_bias_coefficients",
            "posterior_covariance",
            "identifiable_state_transform",
            "identifiable_fractions",
            "query_sensitivity_fractions",
            "robust_weights",
            "anchor_robust_weights",
        )
    }
    return {
        "schema": "bayesian_phystwin.claim_bearing_prob4d_inference_result",
        "schema_version": CLAIM_BEARING_PROB4D_INFERENCE_RESULT_VERSION,
        "inference_admissible": result.inference_admissible,
        "reason": result.reason,
        "arrays": arrays,
        "diagnostics": plain_json(result.diagnostics),
        "input_lineage": plain_json(result.input_lineage),
    }


def _admission_payload(
    *,
    observation_artifact_id: str,
    linearization_artifact_id: str,
    provider_manifest_id: str,
    calibration_artifact_ids: Mapping[str, str],
    runtime_revision_source: str,
    runtime_revision_independently_verified: bool,
    result: GaugeAwareBeliefResult,
) -> dict[str, object]:
    return {
        "schema": "bayesian_phystwin.claim_bearing_prob4d_update",
        "schema_version": CLAIM_BEARING_PROB4D_UPDATE_VERSION,
        "observation_artifact_id": observation_artifact_id,
        "linearization_artifact_id": linearization_artifact_id,
        "provider_manifest_id": provider_manifest_id,
        "calibration_artifact_ids": dict(calibration_artifact_ids),
        "runtime_revision_source": runtime_revision_source,
        "runtime_revision_independently_verified": (
            runtime_revision_independently_verified
        ),
        "inference_admissible": result.inference_admissible,
        "reason": result.reason,
    }


def _bind_claim_bearing_diagnostic_invariants(
    result: GaugeAwareBeliefResult,
) -> GaugeAwareBeliefResult:
    """Record source-prior separation for the strict tree-sparse bridge.

    The tree-sparse solver consumes source-provided reliability and nominal
    probabilities before residual evaluation. Its claim-bearing wrapper binds
    those invariants into the numerical-result identity, matching the established
    dense gauge-aware diagnostics without changing frozen non-tree paths.
    """

    if result.input_lineage.get("prob4d_claim_bearing_tree_sparse_bridge_version") != 1:
        return result
    diagnostics = dict(result.diagnostics)
    for name in (
        "prior_reliability_uses_innovation",
        "prior_nominal_probability_uses_innovation",
    ):
        current = diagnostics.get(name)
        if current not in (None, False):
            raise ValueError(f"claim-bearing result contradicts {name}")
        diagnostics[name] = False
    return replace(result, diagnostics=diagnostics)


@dataclass(frozen=True, slots=True)
class ClaimBearingProb4DUpdateV1:
    """Bound result of strict provider admission and prior-aware inference."""

    result: GaugeAwareBeliefResult
    observation_artifact_id: str
    linearization_artifact_id: str
    provider_manifest_id: str
    calibration_artifact_ids: Mapping[str, str]
    runtime_revision_source: str
    runtime_revision_independently_verified: bool
    _admission_id: str = field(init=False, repr=False, compare=False)
    _inference_result_id: str = field(init=False, repr=False, compare=False)
    _update_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.result, GaugeAwareBeliefResult):
            raise TypeError("result must be a GaugeAwareBeliefResult")
        object.__setattr__(
            self,
            "result",
            _bind_claim_bearing_diagnostic_invariants(self.result),
        )
        for name, value in (
            ("observation_artifact_id", self.observation_artifact_id),
            ("linearization_artifact_id", self.linearization_artifact_id),
            ("provider_manifest_id", self.provider_manifest_id),
        ):
            object.__setattr__(
                self,
                name,
                _validated_sha256(value, name=name),
            )
        calibration_ids = _validated_calibration_ids(self.calibration_artifact_ids)
        runtime_revision_source = _validated_runtime_revision_source(
            self.runtime_revision_source
        )
        if not isinstance(self.runtime_revision_independently_verified, bool):
            raise TypeError("runtime_revision_independently_verified must be a bool")
        if self.runtime_revision_independently_verified is not True:
            raise ValueError("runtime_revision_independently_verified must be True")

        lineage = self.result.input_lineage
        expected = {
            "observation_artifact_id": self.observation_artifact_id,
            "linearization_artifact_id": self.linearization_artifact_id,
            "prob4d_claim_bearing_provider_manifest_id": (self.provider_manifest_id),
            "prob4d_claim_bearing_runtime_revision_source": (runtime_revision_source),
        }
        for key, value in expected.items():
            if lineage.get(key) != value:
                raise ValueError(f"result lineage does not bind {key}")

        lineage_calibration_ids = _validated_calibration_ids(
            lineage.get("prob4d_claim_bearing_calibration_artifact_ids")
        )
        if dict(lineage_calibration_ids) != dict(calibration_ids):
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
            calibration_ids,
        )
        object.__setattr__(
            self,
            "runtime_revision_source",
            runtime_revision_source,
        )

        inference_result_payload = _inference_result_payload(self.result)
        admission_payload = _admission_payload(
            observation_artifact_id=self.observation_artifact_id,
            linearization_artifact_id=self.linearization_artifact_id,
            provider_manifest_id=self.provider_manifest_id,
            calibration_artifact_ids=calibration_ids,
            runtime_revision_source=runtime_revision_source,
            runtime_revision_independently_verified=(
                self.runtime_revision_independently_verified
            ),
            result=self.result,
        )
        admission_id = _canonical_id(admission_payload)
        inference_result_id = _canonical_id(inference_result_payload)
        update_id = _canonical_id(
            {
                **admission_payload,
                "identity_version": CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION,
                "admission_id": admission_id,
                "inference_result_id": inference_result_id,
            }
        )
        object.__setattr__(self, "_admission_id", admission_id)
        object.__setattr__(self, "_inference_result_id", inference_result_id)
        object.__setattr__(self, "_update_id", update_id)

    @property
    def inference_admissible(self) -> bool:
        return self.result.inference_admissible

    @property
    def admission_id(self) -> str:
        """Return the historical provenance-and-decision identity."""

        return self._admission_id

    @property
    def legacy_update_id(self) -> str:
        """Backward-compatible name for the pre-hardening update identity."""

        return self._admission_id

    @property
    def inference_result_id(self) -> str:
        return self._inference_result_id

    @property
    def update_id(self) -> str:
        """Bind provider admission and the complete numerical inference result."""

        return self._update_id


@dataclass(frozen=True, slots=True)
class ClaimBearingProb4DCandidateV1:
    """Covariance-typed candidate belief before complete deployment selection.

    This contract wraps the frozen V1 update identity rather than replacing it.
    An accepted strict solve carries working IRLS covariance semantics. A rejected
    solve carries exact-prior-fallback semantics. Neither path is a deployment
    decision; nonlinear closure, the source-frozen guard, and complete-belief
    selection remain separate.
    """

    update_v1: ClaimBearingProb4DUpdateV1
    covariance_semantics: PosteriorCovarianceSemanticsV1
    _candidate_result_id: str = field(init=False, repr=False, compare=False)
    _candidate_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.update_v1, ClaimBearingProb4DUpdateV1):
            raise TypeError("update_v1 must be a ClaimBearingProb4DUpdateV1")
        semantics = self.covariance_semantics
        if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
            raise TypeError(
                "covariance_semantics must be a PosteriorCovarianceSemanticsV1"
            )
        dimension = self.result.posterior_covariance.shape[0]
        if semantics.dimension != dimension:
            raise ValueError(
                "covariance_semantics dimension must match posterior covariance"
            )
        expected_method = (
            "irls_working" if self.inference_admissible else "exact_prior_fallback"
        )
        if semantics.method != expected_method:
            raise ValueError(
                "covariance_semantics method contradicts the admission decision"
            )
        if self.inference_admissible:
            if not semantics.prior_included or not semantics.generalized_bayes:
                raise ValueError(
                    "working IRLS semantics contradict the claim-bearing solver"
                )
        elif semantics.metadata.get("fallback_reason") != self.reason:
            raise ValueError(
                "exact-prior semantics do not bind the rejected result reason"
            )
        if semantics.calibrated:
            raise ValueError(
                "claim-bearing candidate covariance must remain explicitly raw"
            )

        result_payload = {
            "schema": "bayesian_phystwin.claim_bearing_prob4d_candidate_result",
            "schema_version": CLAIM_BEARING_PROB4D_CANDIDATE_RESULT_VERSION,
            "v1_inference_result_id": self.update_v1.inference_result_id,
            "covariance_semantics_id": semantics.artifact_id,
        }
        candidate_result_id = _canonical_id(result_payload)
        object.__setattr__(self, "_candidate_result_id", candidate_result_id)
        object.__setattr__(self, "covariance_semantics", semantics)
        object.__setattr__(self, "_candidate_id", _canonical_id(self.descriptor()))

    @property
    def result(self) -> GaugeAwareBeliefResult:
        return self.update_v1.result

    @property
    def inference_admissible(self) -> bool:
        return self.update_v1.inference_admissible

    @property
    def reason(self) -> str:
        return self.result.reason

    @property
    def admission_id(self) -> str:
        return self.update_v1.admission_id

    @property
    def v1_update_id(self) -> str:
        return self.update_v1.update_id

    @property
    def v1_inference_result_id(self) -> str:
        return self.update_v1.inference_result_id

    @property
    def candidate_result_id(self) -> str:
        return self._candidate_result_id

    @property
    def candidate_id(self) -> str:
        return self._candidate_id

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": "bayesian_phystwin.claim_bearing_prob4d_candidate",
            "schema_version": CLAIM_BEARING_PROB4D_CANDIDATE_VERSION,
            "identity_version": CLAIM_BEARING_PROB4D_CANDIDATE_IDENTITY_VERSION,
            "admission_id": self.admission_id,
            "v1_update_id": self.v1_update_id,
            "v1_inference_result_id": self.v1_inference_result_id,
            "candidate_result_id": self.candidate_result_id,
            "covariance_semantics_id": self.covariance_semantics.artifact_id,
            "inference_admissible": self.inference_admissible,
            "reason": self.reason,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "covariance_semantics": self.covariance_semantics.to_record(),
            "candidate_id": self.candidate_id,
        }


def bind_claim_bearing_prob4d_candidate(
    update: ClaimBearingProb4DUpdateV1,
    *,
    covariance_semantics: PosteriorCovarianceSemanticsV1 | None = None,
) -> ClaimBearingProb4DCandidateV1:
    """Bind typed covariance meaning to one frozen V1 inference result."""

    if not isinstance(update, ClaimBearingProb4DUpdateV1):
        raise TypeError("update must be a ClaimBearingProb4DUpdateV1")
    semantics = covariance_semantics
    if semantics is None:
        metadata = {"source": "claim-bearing-prob4d-strict-v2"}
        if update.inference_admissible:
            semantics = working_irls_covariance_semantics(
                update.result.posterior_covariance,
                metadata=metadata,
            )
        else:
            semantics = exact_prior_fallback_covariance_semantics(
                update.result.posterior_covariance,
                reason=update.result.reason,
                metadata=metadata,
            )
    return ClaimBearingProb4DCandidateV1(
        update_v1=update,
        covariance_semantics=semantics,
    )


def update_claim_bearing_prob4d_from_artifacts(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    **anchor_dependence: Any,
) -> ClaimBearingProb4DUpdateV1:
    """Validate stream-v2 evidence before forming and solving the update.

    This is the frozen V1 one-call composition. It deliberately uses strict-v2
    admission around the historical prior-aware grouped-mixture solver. New
    prospective callers should use
    ``infer_claim_bearing_prob4d_candidate_from_artifacts`` so the covariance
    interpretation is bound before complete-belief selection.
    """

    adapted = build_claim_bearing_gauge_aware_batch_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        **anchor_dependence,
    )
    metadata = adapted.batch.metadata or {}
    provider_manifest_id = _validated_sha256(
        metadata.get("prob4d_claim_bearing_provider_manifest_id"),
        name="provider_manifest_id",
    )
    calibration_ids = _validated_calibration_ids(
        metadata.get("prob4d_claim_bearing_calibration_artifact_ids")
    )
    runtime_revision_source = _validated_runtime_revision_source(
        metadata.get("prob4d_claim_bearing_runtime_revision_source")
    )
    runtime_verified = metadata.get(
        "prob4d_claim_bearing_runtime_revision_independently_verified"
    )
    if runtime_verified is not True:
        raise ValueError(
            "claim-bearing Prob4D runtime revision was not independently verified"
        )
    result = update_prior_aware_gauge_belief(
        adapted.batch,
        config=config,
    )
    return ClaimBearingProb4DUpdateV1(
        result=result,
        observation_artifact_id=adapted.observation_artifact_id,
        linearization_artifact_id=linearization.artifact_id,
        provider_manifest_id=provider_manifest_id,
        calibration_artifact_ids=calibration_ids,
        runtime_revision_source=runtime_revision_source,
        runtime_revision_independently_verified=True,
    )


def infer_claim_bearing_prob4d_candidate_from_artifacts(
    observation_belief: ObservationBeliefV1,
    linearization: PhysicalLinearizationV1,
    *,
    physical_prediction_xyz_m: np.ndarray,
    shared_bias_jacobian: np.ndarray | None = None,
    view_bias_jacobian: np.ndarray | None = None,
    state_prior_covariance_m2: np.ndarray | None = None,
    anchor_innovation_m: np.ndarray | None = None,
    anchor_covariance_m2: np.ndarray | None = None,
    anchor_state_jacobian: np.ndarray | None = None,
    config: PriorAwareGaugeConfigV1 | None = None,
    covariance_semantics: PosteriorCovarianceSemanticsV1 | None = None,
    **anchor_dependence: Any,
) -> ClaimBearingProb4DCandidateV1:
    """Infer a covariance-typed candidate without making a deployment decision."""

    update = update_claim_bearing_prob4d_from_artifacts(
        observation_belief,
        linearization,
        physical_prediction_xyz_m=physical_prediction_xyz_m,
        shared_bias_jacobian=shared_bias_jacobian,
        view_bias_jacobian=view_bias_jacobian,
        state_prior_covariance_m2=state_prior_covariance_m2,
        anchor_innovation_m=anchor_innovation_m,
        anchor_covariance_m2=anchor_covariance_m2,
        anchor_state_jacobian=anchor_state_jacobian,
        config=config,
        **anchor_dependence,
    )
    return bind_claim_bearing_prob4d_candidate(
        update,
        covariance_semantics=covariance_semantics,
    )


__all__ = [
    "CLAIM_BEARING_PROB4D_CANDIDATE_IDENTITY_VERSION",
    "CLAIM_BEARING_PROB4D_CANDIDATE_RESULT_VERSION",
    "CLAIM_BEARING_PROB4D_CANDIDATE_VERSION",
    "CLAIM_BEARING_PROB4D_INFERENCE_RESULT_VERSION",
    "CLAIM_BEARING_PROB4D_UPDATE_IDENTITY_VERSION",
    "CLAIM_BEARING_PROB4D_UPDATE_VERSION",
    "ClaimBearingProb4DCandidateV1",
    "ClaimBearingProb4DUpdateV1",
    "bind_claim_bearing_prob4d_candidate",
    "infer_claim_bearing_prob4d_candidate_from_artifacts",
    "update_claim_bearing_prob4d_from_artifacts",
]
