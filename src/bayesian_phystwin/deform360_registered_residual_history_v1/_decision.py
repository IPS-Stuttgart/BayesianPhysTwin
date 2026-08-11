"""Content-addressed acceptance and exact-reference fallback records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._canonical_contracts import frozen_finite_json_mapping, plain_json
from .._portable_contracts import content_id
from ..contracts.fixed_anchor import FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION
from ..covariance_only_hybrid import CovarianceOnlyHybridPredictionV1
from ..endpoint_model_average import MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION
from ._common import (
    _ALLOWED_FALLBACK_REASONS,
    CLAIM_BOUNDARY,
    REGISTERED_COVARIANCE_DONOR_ID,
    REGISTERED_COVARIANCE_SCALES,
    REGISTERED_DECISION_SCHEMA,
    REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK,
    REGISTERED_REFERENCE_PREDICTOR_ID,
    REGISTERED_SCHEMA_VERSION,
    _array_sha256,
    _canonical_horizon_bins,
    _canonical_string,
    _float64_array,
    _frozen_endpoint_config_id,
    _integer,
    _optional_sha256,
    _sha256,
)
from ._provenance import ResidualHistorySourceProvenanceV1


@dataclass(frozen=True, slots=True)
class RegisteredResidualHistoryDecisionV1:
    """Content-addressed acceptance or exact registered-reference fallback."""

    source_unit_id: str
    provenance_id: str
    residual_history_sha256: str
    validity_sha256: str
    registered_mean_sha256: str
    reconstructed_reference_mean_sha256: str
    reference_covariance_sha256: str
    valid_observation_count_by_track: tuple[int, ...]
    future_horizon_count: int
    future_horizon_bins: tuple[int, ...]
    future_horizon_steps: tuple[int, ...]
    scale_schedule_sha256: str
    endpoint_contract_version: int
    fixed_anchor_contract_version: int
    endpoint_config_id: str
    accepted: bool
    fallback_reasons: tuple[str, ...]
    endpoint_posterior_id: str | None
    endpoint_prediction_ids: tuple[str, ...]
    donor_covariance_sha256: str | None
    output_covariance_sha256: str
    hybrid_artifact_id: str | None
    registered_mean_identity_preserved: bool
    reference_covariance_identity_preserved: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str | None = None

    def __post_init__(self) -> None:
        source_unit = _canonical_string(self.source_unit_id, name="source_unit_id")
        provenance_id = _sha256(self.provenance_id, name="provenance_id")
        digest_names = (
            "residual_history_sha256",
            "validity_sha256",
            "registered_mean_sha256",
            "reconstructed_reference_mean_sha256",
            "reference_covariance_sha256",
            "scale_schedule_sha256",
            "output_covariance_sha256",
        )
        digests = {
            name: _sha256(getattr(self, name), name=name) for name in digest_names
        }
        counts = self._validated_counts()
        future_count = _integer(
            self.future_horizon_count,
            name="future_horizon_count",
            minimum=3,
        )
        bins = self._validated_bins(future_count)
        steps = self._validated_steps(future_count)
        endpoint_version = _integer(
            self.endpoint_contract_version,
            name="endpoint_contract_version",
            minimum=1,
        )
        fixed_anchor_version = _integer(
            self.fixed_anchor_contract_version,
            name="fixed_anchor_contract_version",
            minimum=1,
        )
        if endpoint_version != MODEL_AVERAGED_ENDPOINT_CONTRACT_VERSION:
            raise ValueError("endpoint contract version changed")
        if fixed_anchor_version != FIXED_BAYESIAN_ANCHOR_CONTRACT_VERSION:
            raise ValueError("fixed-anchor contract version changed")
        config_id = _sha256(self.endpoint_config_id, name="endpoint_config_id")
        if config_id != _frozen_endpoint_config_id():
            raise ValueError("endpoint configuration identity changed")
        reasons = self._validated_reasons(counts, digests)
        posterior_id = _optional_sha256(
            self.endpoint_posterior_id,
            name="endpoint_posterior_id",
        )
        prediction_ids = self._validated_prediction_ids()
        donor_sha = _optional_sha256(
            self.donor_covariance_sha256,
            name="donor_covariance_sha256",
        )
        hybrid_id = _optional_sha256(
            self.hybrid_artifact_id,
            name="hybrid_artifact_id",
        )
        donor_empty = posterior_id is None and not prediction_ids and donor_sha is None
        donor_complete = (
            posterior_id is not None
            and len(prediction_ids) == future_count
            and donor_sha is not None
        )
        if not (donor_empty or donor_complete):
            raise ValueError("endpoint donor lineage must be complete or absent")
        self._validate_decision(
            counts=counts,
            digests=digests,
            reasons=reasons,
            donor_empty=donor_empty,
            donor_complete=donor_complete,
            hybrid_id=hybrid_id,
        )
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        assignments = {
            "source_unit_id": source_unit,
            "provenance_id": provenance_id,
            **digests,
            "valid_observation_count_by_track": counts,
            "future_horizon_count": future_count,
            "future_horizon_bins": bins,
            "future_horizon_steps": steps,
            "endpoint_contract_version": endpoint_version,
            "fixed_anchor_contract_version": fixed_anchor_version,
            "endpoint_config_id": config_id,
            "fallback_reasons": reasons,
            "endpoint_posterior_id": posterior_id,
            "endpoint_prediction_ids": prediction_ids,
            "donor_covariance_sha256": donor_sha,
            "hybrid_artifact_id": hybrid_id,
            "metadata": metadata,
        }
        for name, value in assignments.items():
            object.__setattr__(self, name, value)
        expected = content_id(self.descriptor())
        if self.decision_id is None:
            object.__setattr__(self, "decision_id", expected)
        elif _sha256(self.decision_id, name="decision_id") != expected:
            raise ValueError("decision_id does not match the registered decision")

    def _validated_counts(self) -> tuple[int, ...]:
        if type(self.valid_observation_count_by_track) is not tuple:
            raise ValueError(
                "valid_observation_count_by_track must be a canonical tuple"
            )
        counts = tuple(
            _integer(value, name=f"count[{index}]", minimum=0)
            for index, value in enumerate(self.valid_observation_count_by_track)
        )
        if not counts:
            raise ValueError("valid observation counts must be nonempty")
        return counts

    def _validated_bins(self, future_count: int) -> tuple[int, ...]:
        if type(self.future_horizon_bins) is not tuple:
            raise ValueError("future_horizon_bins must be a canonical tuple")
        expected = tuple(int(value) for value in _canonical_horizon_bins(future_count))
        if self.future_horizon_bins != expected:
            raise ValueError("future_horizon_bins differ from the canonical partition")
        return expected

    def _validated_steps(self, future_count: int) -> tuple[int, ...]:
        if type(self.future_horizon_steps) is not tuple:
            raise ValueError("future_horizon_steps must be a canonical tuple")
        expected = tuple(range(1, future_count + 1))
        if self.future_horizon_steps != expected:
            raise ValueError("future_horizon_steps must be consecutive and complete")
        return expected

    def _validated_reasons(
        self,
        counts: tuple[int, ...],
        digests: Mapping[str, str],
    ) -> tuple[str, ...]:
        if type(self.fallback_reasons) is not tuple:
            raise ValueError("fallback_reasons must be a canonical tuple")
        reasons = tuple(
            _canonical_string(value, name="fallback_reasons")
            for value in self.fallback_reasons
        )
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("fallback_reasons must be sorted and unique")
        if not set(reasons) <= _ALLOWED_FALLBACK_REASONS:
            raise ValueError("fallback_reasons contain an unsupported reason")
        insufficient = any(
            count < REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK for count in counts
        )
        if ("insufficient-per-track-support" in reasons) != insufficient:
            raise ValueError("support fallback reason differs from source support")
        mean_mismatch = (
            digests["registered_mean_sha256"]
            != digests["reconstructed_reference_mean_sha256"]
        )
        if ("registered-mean-mismatch" in reasons) != mean_mismatch:
            raise ValueError("mean mismatch fallback reason differs from content")
        if "covariance-contract-rejection" in reasons and len(reasons) != 1:
            raise ValueError("covariance rejection must be the sole fallback reason")
        return reasons

    def _validated_prediction_ids(self) -> tuple[str, ...]:
        if type(self.endpoint_prediction_ids) is not tuple:
            raise ValueError("endpoint_prediction_ids must be a canonical tuple")
        result = tuple(
            _sha256(value, name=f"endpoint_prediction_ids[{index}]")
            for index, value in enumerate(self.endpoint_prediction_ids)
        )
        if len(set(result)) != len(result):
            raise ValueError("endpoint_prediction_ids must be unique")
        return result

    def _validate_decision(
        self,
        *,
        counts: tuple[int, ...],
        digests: Mapping[str, str],
        reasons: tuple[str, ...],
        donor_empty: bool,
        donor_complete: bool,
        hybrid_id: str | None,
    ) -> None:
        identity_values = (
            self.registered_mean_identity_preserved,
            self.reference_covariance_identity_preserved,
        )
        if type(self.accepted) is not bool or any(
            type(value) is not bool for value in identity_values
        ):
            raise ValueError("decision and identity fields must be Booleans")
        if not self.registered_mean_identity_preserved:
            raise ValueError("registered mean identity must always be preserved")
        insufficient = any(
            count < REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK for count in counts
        )
        mean_mismatch = (
            digests["registered_mean_sha256"]
            != digests["reconstructed_reference_mean_sha256"]
        )
        if self.accepted:
            if reasons or insufficient or mean_mismatch:
                raise ValueError("accepted decision violates registered admission")
            if not donor_complete or hybrid_id is None:
                raise ValueError("accepted decision lacks endpoint donor lineage")
            if self.reference_covariance_identity_preserved:
                raise ValueError("accepted decision retained reference covariance")
            return
        if not reasons:
            raise ValueError("fallback decisions require at least one reason")
        if hybrid_id is not None:
            raise ValueError("fallback decision must not retain a covariance hybrid")
        if not self.reference_covariance_identity_preserved:
            raise ValueError("fallback must preserve reference covariance identity")
        if (
            digests["output_covariance_sha256"]
            != digests["reference_covariance_sha256"]
        ):
            raise ValueError("fallback covariance differs from the reference")
        covariance_rejection = reasons == ("covariance-contract-rejection",)
        if covariance_rejection:
            if not (donor_empty or donor_complete):
                raise ValueError("covariance rejection has partial donor lineage")
        elif not donor_empty:
            raise ValueError("admission fallback must not retain donor execution")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": REGISTERED_DECISION_SCHEMA,
            "schema_version": REGISTERED_SCHEMA_VERSION,
            "source_unit_id": self.source_unit_id,
            "provenance_id": self.provenance_id,
            "residual_history_sha256": self.residual_history_sha256,
            "validity_sha256": self.validity_sha256,
            "registered_mean_sha256": self.registered_mean_sha256,
            "reconstructed_reference_mean_sha256": (
                self.reconstructed_reference_mean_sha256
            ),
            "reference_covariance_sha256": self.reference_covariance_sha256,
            "valid_observation_count_by_track": list(
                self.valid_observation_count_by_track
            ),
            "minimum_valid_observations_per_track": (
                REGISTERED_MINIMUM_VALID_OBSERVATIONS_PER_TRACK
            ),
            "future_horizon_count": self.future_horizon_count,
            "future_horizon_bins": list(self.future_horizon_bins),
            "future_horizon_steps": list(self.future_horizon_steps),
            "registered_covariance_scales": list(REGISTERED_COVARIANCE_SCALES),
            "scale_schedule_sha256": self.scale_schedule_sha256,
            "endpoint_contract_version": self.endpoint_contract_version,
            "fixed_anchor_contract_version": self.fixed_anchor_contract_version,
            "endpoint_config_id": self.endpoint_config_id,
            "accepted": self.accepted,
            "fallback_reasons": list(self.fallback_reasons),
            "endpoint_posterior_id": self.endpoint_posterior_id,
            "endpoint_prediction_ids": list(self.endpoint_prediction_ids),
            "donor_covariance_sha256": self.donor_covariance_sha256,
            "output_covariance_sha256": self.output_covariance_sha256,
            "hybrid_artifact_id": self.hybrid_artifact_id,
            "registered_mean_identity_preserved": (
                self.registered_mean_identity_preserved
            ),
            "reference_covariance_identity_preserved": (
                self.reference_covariance_identity_preserved
            ),
            "reference_predictor_id": REGISTERED_REFERENCE_PREDICTOR_ID,
            "covariance_donor_id": REGISTERED_COVARIANCE_DONOR_ID,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class RegisteredResidualHistoryPredictionV1:
    """One source-only candidate or exact registered-reference fallback."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    provenance: ResidualHistorySourceProvenanceV1
    decision: RegisteredResidualHistoryDecisionV1
    hybrid: CovarianceOnlyHybridPredictionV1 | None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ResidualHistorySourceProvenanceV1):
            raise TypeError("provenance must be ResidualHistorySourceProvenanceV1")
        if not isinstance(self.decision, RegisteredResidualHistoryDecisionV1):
            raise TypeError("decision must be RegisteredResidualHistoryDecisionV1")
        if self.decision.provenance_id != self.provenance.provenance_id:
            raise ValueError("decision and source provenance differ")
        mean = _float64_array(
            self.mean_m,
            name="mean_m",
            ndim=3,
            require_finite=True,
        )
        covariance = _float64_array(
            self.covariance_m2,
            name="covariance_m2",
            ndim=4,
            require_finite=True,
        )
        expected_shape = (
            self.decision.future_horizon_count,
            len(self.decision.valid_observation_count_by_track),
            3,
        )
        if mean.shape != expected_shape:
            raise ValueError("result mean shape differs from the registered decision")
        if covariance.shape != expected_shape + (3,):
            raise ValueError("result covariance shape differs from the decision")
        if _array_sha256(mean) != self.decision.registered_mean_sha256:
            raise ValueError("result mean content differs from the registered mean")
        if _array_sha256(covariance) != self.decision.output_covariance_sha256:
            raise ValueError("result covariance content differs from the decision")
        if self.decision.accepted:
            if self.hybrid is None:
                raise ValueError("accepted result is missing the covariance hybrid")
            if (
                self.hybrid.mean_m is not mean
                or self.hybrid.covariance_m2 is not covariance
            ):
                raise ValueError("accepted result does not retain hybrid objects")
        elif self.hybrid is not None:
            raise ValueError("fallback result must not retain a covariance hybrid")

    @property
    def accepted(self) -> bool:
        return self.decision.accepted
