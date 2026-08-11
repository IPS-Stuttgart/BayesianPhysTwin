"""Admission, covariance composition, and exact fallback for the dry run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._deform360_covariance_residual_history_adapter_v1 import (
    ResidualHistoryAdapterV1,
    build_residual_history_adapter,
)
from ._deform360_covariance_residual_history_common_v1 import (
    CLAIM_BOUNDARY,
    FALLBACK_SEMANTICS,
    HORIZON_LABELS,
    REFERENCE_MEAN_SEMANTICS,
    RESIDUAL_HISTORY_DECISION_SCHEMA,
    SCHEMA_VERSION,
    ResidualHistoryDryRunPolicyV1,
    _array_sha256,
    _canonical_string,
    _finite_real,
    _integer_vector,
    _required_sha256,
    _validate_covariance,
)
from ._portable_contracts import content_id
from .covariance_only_hybrid import (
    CovarianceOnlyHybridPredictionV1,
    compose_covariance_only_hybrid,
)


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunDecisionV1:
    """Content-addressed admission or exact-fallback decision."""

    source_unit_id: str
    adapter_id: str
    policy_id: str
    partition_id: str
    accepted: bool
    fallback_reasons: tuple[str, ...]
    final_observed_count: int
    final_observed_fraction: float
    future_horizon_bins_sha256: str
    physical_future_mean_sha256: str
    physical_fallback_covariance_sha256: str
    deployed_mean_sha256: str
    deployed_covariance_sha256: str
    hybrid_artifact_id: str | None
    hybrid_reference_mean_identity_preserved: bool
    exact_physical_fallback_mean_identity_preserved: bool
    exact_physical_fallback_covariance_identity_preserved: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decision_id: str | None = None

    def __post_init__(self) -> None:
        source_unit_id = _canonical_string(self.source_unit_id, name="source_unit_id")
        adapter_id = literal_lower_hex(self.adapter_id, name="adapter_id", lengths={64})
        policy_id = literal_lower_hex(self.policy_id, name="policy_id", lengths={64})
        partition_id = literal_lower_hex(
            self.partition_id,
            name="partition_id",
            lengths={64},
        )
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a Boolean")
        if type(self.fallback_reasons) is not tuple:
            raise ValueError("fallback_reasons must be a canonical tuple")
        reasons = tuple(
            _canonical_string(value, name="fallback_reasons")
            for value in self.fallback_reasons
        )
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("fallback_reasons must be sorted and unique")
        final_count = genuine_integer(
            self.final_observed_count,
            name="final_observed_count",
            minimum=0,
        )
        fraction = _finite_real(
            self.final_observed_fraction,
            name="final_observed_fraction",
            minimum=0.0,
        )
        if fraction > 1.0:
            raise ValueError("final_observed_fraction must not exceed one")
        digests = {}
        for name in (
            "future_horizon_bins_sha256",
            "physical_future_mean_sha256",
            "physical_fallback_covariance_sha256",
            "deployed_mean_sha256",
            "deployed_covariance_sha256",
        ):
            digests[name] = literal_lower_hex(
                getattr(self, name),
                name=name,
                lengths={64},
            )
        hybrid_id = self.hybrid_artifact_id
        if hybrid_id is not None:
            hybrid_id = literal_lower_hex(
                hybrid_id,
                name="hybrid_artifact_id",
                lengths={64},
            )
        boolean_fields = (
            "hybrid_reference_mean_identity_preserved",
            "exact_physical_fallback_mean_identity_preserved",
            "exact_physical_fallback_covariance_identity_preserved",
        )
        if any(type(getattr(self, name)) is not bool for name in boolean_fields):
            raise ValueError("identity-preservation fields must be Booleans")
        if self.accepted:
            if reasons or hybrid_id is None:
                raise ValueError(
                    "accepted decisions require one hybrid and no fallback"
                )
            if not self.hybrid_reference_mean_identity_preserved:
                raise ValueError(
                    "accepted hybrid must preserve its reference mean identity"
                )
            if (
                self.exact_physical_fallback_mean_identity_preserved
                or self.exact_physical_fallback_covariance_identity_preserved
            ):
                raise ValueError("accepted decision must not claim fallback identity")
        else:
            if not reasons or hybrid_id is not None:
                raise ValueError("fallback decisions require reasons and no hybrid")
            if self.hybrid_reference_mean_identity_preserved:
                raise ValueError("fallback decision has no hybrid reference")
            if not (
                self.exact_physical_fallback_mean_identity_preserved
                and self.exact_physical_fallback_covariance_identity_preserved
            ):
                raise ValueError(
                    "fallback must preserve both physical objects by identity"
                )
        metadata = frozen_finite_json_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "source_unit_id", source_unit_id)
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "partition_id", partition_id)
        object.__setattr__(self, "fallback_reasons", reasons)
        object.__setattr__(self, "final_observed_count", final_count)
        object.__setattr__(self, "final_observed_fraction", fraction)
        for name, value in digests.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "hybrid_artifact_id", hybrid_id)
        object.__setattr__(self, "metadata", metadata)
        expected = content_id(self.descriptor())
        if self.decision_id is None:
            object.__setattr__(self, "decision_id", expected)
        elif (
            literal_lower_hex(
                self.decision_id,
                name="decision_id",
                lengths={64},
            )
            != expected
        ):
            raise ValueError("decision_id does not match the dry-run decision")

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": RESIDUAL_HISTORY_DECISION_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_unit_id": self.source_unit_id,
            "adapter_id": self.adapter_id,
            "policy_id": self.policy_id,
            "partition_id": self.partition_id,
            "accepted": self.accepted,
            "fallback_reasons": list(self.fallback_reasons),
            "final_observed_count": self.final_observed_count,
            "final_observed_fraction": self.final_observed_fraction,
            "future_horizon_bins_sha256": self.future_horizon_bins_sha256,
            "physical_future_mean_sha256": self.physical_future_mean_sha256,
            "physical_fallback_covariance_sha256": (
                self.physical_fallback_covariance_sha256
            ),
            "deployed_mean_sha256": self.deployed_mean_sha256,
            "deployed_covariance_sha256": self.deployed_covariance_sha256,
            "hybrid_artifact_id": self.hybrid_artifact_id,
            "hybrid_reference_mean_identity_preserved": (
                self.hybrid_reference_mean_identity_preserved
            ),
            "exact_physical_fallback_mean_identity_preserved": (
                self.exact_physical_fallback_mean_identity_preserved
            ),
            "exact_physical_fallback_covariance_identity_preserved": (
                self.exact_physical_fallback_covariance_identity_preserved
            ),
            "reference_mean_semantics": REFERENCE_MEAN_SEMANTICS,
            "fallback_semantics": FALLBACK_SEMANTICS,
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class ResidualHistoryDryRunResultV1:
    """One source-only dry-run result and its runtime identity proofs."""

    mean_m: np.ndarray
    covariance_m2: np.ndarray
    adapter: ResidualHistoryAdapterV1
    decision: ResidualHistoryDryRunDecisionV1
    hybrid: CovarianceOnlyHybridPredictionV1 | None

    @property
    def accepted(self) -> bool:
        return self.decision.accepted


def _physical_future_mean(value: object, *, material_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("physical_future_m must be a NumPy array to preserve identity")
    if value.dtype != np.dtype(np.float64):
        raise ValueError("physical_future_m must have dtype float64")
    if value.ndim != 3 or value.shape[1:] != (material_count, 3) or value.shape[0] < 1:
        raise ValueError("physical_future_m must have shape (H, N, 3)")
    if not value.flags.c_contiguous:
        raise ValueError("physical_future_m must be C-contiguous")
    if not np.all(np.isfinite(value)):
        raise ValueError("physical_future_m must be finite")
    return value


def _horizon_bins(value: object, *, future_count: int) -> np.ndarray:
    bins = _integer_vector(value, name="future_horizon_bins")
    if bins.shape != (future_count,):
        raise ValueError("future_horizon_bins must have one entry per future frame")
    if np.any((bins < 0) | (bins >= len(HORIZON_LABELS))):
        raise ValueError(
            "future_horizon_bins must use early/middle/late indices 0, 1, 2"
        )
    return bins


def _fallback_result(
    *,
    physical_future: np.ndarray,
    physical_covariance: np.ndarray,
    adapter: ResidualHistoryAdapterV1,
    policy: ResidualHistoryDryRunPolicyV1,
    horizon_bins: np.ndarray,
    reasons: Sequence[str],
    metadata: Mapping[str, Any] | None,
) -> ResidualHistoryDryRunResultV1:
    canonical_reasons = tuple(sorted(set(reasons)))
    adapter_id = _required_sha256(adapter.adapter_id, name="adapter_id")
    partition_id = _required_sha256(
        adapter.partition.partition_id,
        name="partition_id",
    )
    decision = ResidualHistoryDryRunDecisionV1(
        source_unit_id=adapter.source_unit_id,
        adapter_id=adapter_id,
        policy_id=_required_sha256(policy.policy_id, name="policy_id"),
        partition_id=partition_id,
        accepted=False,
        fallback_reasons=canonical_reasons,
        final_observed_count=adapter.final_observed_count,
        final_observed_fraction=adapter.final_observed_fraction,
        future_horizon_bins_sha256=_array_sha256(horizon_bins),
        physical_future_mean_sha256=_array_sha256(physical_future),
        physical_fallback_covariance_sha256=_array_sha256(physical_covariance),
        deployed_mean_sha256=_array_sha256(physical_future),
        deployed_covariance_sha256=_array_sha256(physical_covariance),
        hybrid_artifact_id=None,
        hybrid_reference_mean_identity_preserved=False,
        exact_physical_fallback_mean_identity_preserved=True,
        exact_physical_fallback_covariance_identity_preserved=True,
        metadata={} if metadata is None else metadata,
    )
    result = ResidualHistoryDryRunResultV1(
        mean_m=physical_future,
        covariance_m2=physical_covariance,
        adapter=adapter,
        decision=decision,
        hybrid=None,
    )
    if (
        result.mean_m is not physical_future
        or result.covariance_m2 is not physical_covariance
    ):
        raise AssertionError(
            "exact physical fallback object identity was not preserved"
        )
    return result


def run_source_only_residual_history_dry_run(
    physical_prefix_m: object,
    provider_observation_prefix_m: object,
    observed_validity: object,
    physical_future_m: np.ndarray,
    physical_fallback_covariance_m2: np.ndarray,
    donor_covariance_m2: object,
    *,
    frame_indices: object,
    material_ids: object,
    future_horizon_bins: object,
    camera_ids: Sequence[str],
    provider_camera_ids: Sequence[str],
    scoring_camera_ids: Sequence[str],
    provider_reconstruction_artifact_id: str,
    scoring_reconstruction_artifact_id: str,
    source_unit_id: str,
    reference_predictor_id: str,
    covariance_donor_id: str,
    policy: ResidualHistoryDryRunPolicyV1,
    metadata: Mapping[str, Any] | None = None,
) -> ResidualHistoryDryRunResultV1:
    """Run the no-fill source gate and deploy either hybrid or exact fallback."""

    adapter = build_residual_history_adapter(
        physical_prefix_m,
        provider_observation_prefix_m,
        observed_validity,
        frame_indices=frame_indices,
        material_ids=material_ids,
        camera_ids=camera_ids,
        provider_camera_ids=provider_camera_ids,
        scoring_camera_ids=scoring_camera_ids,
        provider_reconstruction_artifact_id=provider_reconstruction_artifact_id,
        scoring_reconstruction_artifact_id=scoring_reconstruction_artifact_id,
        source_unit_id=source_unit_id,
        policy=policy,
        metadata=metadata,
    )
    physical_future = _physical_future_mean(
        physical_future_m,
        material_count=adapter.material_count,
    )
    future_shape = physical_future.shape
    physical_covariance = _validate_covariance(
        physical_fallback_covariance_m2,
        name="physical_fallback_covariance_m2",
        expected_shape=future_shape + (3,),
        preserve_identity=True,
    )
    bins = _horizon_bins(future_horizon_bins, future_count=future_shape[0])
    reasons: list[str] = []
    if adapter.final_observed_count < policy.minimum_final_observed_count:
        reasons.append("minimum-final-observed-count")
    if adapter.final_observed_fraction < policy.minimum_final_observed_fraction:
        reasons.append("minimum-final-observed-fraction")
    if reasons:
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            adapter=adapter,
            policy=policy,
            horizon_bins=bins,
            reasons=reasons,
            metadata=metadata,
        )

    final_support = np.asarray(adapter.observed_validity[-1], dtype=bool)
    reference_mean = np.array(physical_future, dtype=np.float64, copy=True, order="C")
    reference_mean[:, final_support] += np.asarray(
        adapter.residual_history_m[-1, final_support],
        dtype=np.float64,
    )[None, ...]
    scales = np.asarray(policy.covariance_scales, dtype=np.float64)[bins]
    try:
        hybrid = compose_covariance_only_hybrid(
            reference_mean,
            donor_covariance_m2,
            reference_predictor_id=_canonical_string(
                reference_predictor_id,
                name="reference_predictor_id",
            ),
            covariance_donor_id=_canonical_string(
                covariance_donor_id,
                name="covariance_donor_id",
            ),
            covariance_scale=scales[:, None],
            metadata={
                "source_unit_id": adapter.source_unit_id,
                "adapter_id": _required_sha256(
                    adapter.adapter_id,
                    name="adapter_id",
                ),
                "policy_id": policy.policy_id,
                "partition_id": _required_sha256(
                    adapter.partition.partition_id,
                    name="partition_id",
                ),
                "future_horizon_labels": [HORIZON_LABELS[int(value)] for value in bins],
            },
        )
    except (TypeError, ValueError) as error:
        fallback_metadata: dict[str, Any] = {
            "covariance_rejection_type": type(error).__name__,
            "covariance_rejection_message": str(error),
        }
        if metadata is not None:
            fallback_metadata["dry_run_metadata"] = plain_json(metadata)
        return _fallback_result(
            physical_future=physical_future,
            physical_covariance=physical_covariance,
            adapter=adapter,
            policy=policy,
            horizon_bins=bins,
            reasons=("covariance-contract-rejection",),
            metadata=fallback_metadata,
        )
    if hybrid.mean_m is not reference_mean:
        raise AssertionError("covariance-only helper copied the last-residual mean")
    adapter_id = _required_sha256(adapter.adapter_id, name="adapter_id")
    partition_id = _required_sha256(
        adapter.partition.partition_id,
        name="partition_id",
    )
    hybrid_id = _required_sha256(
        hybrid.record.artifact_id,
        name="hybrid_artifact_id",
    )
    decision = ResidualHistoryDryRunDecisionV1(
        source_unit_id=adapter.source_unit_id,
        adapter_id=adapter_id,
        policy_id=_required_sha256(policy.policy_id, name="policy_id"),
        partition_id=partition_id,
        accepted=True,
        fallback_reasons=(),
        final_observed_count=adapter.final_observed_count,
        final_observed_fraction=adapter.final_observed_fraction,
        future_horizon_bins_sha256=_array_sha256(bins),
        physical_future_mean_sha256=_array_sha256(physical_future),
        physical_fallback_covariance_sha256=_array_sha256(physical_covariance),
        deployed_mean_sha256=_array_sha256(hybrid.mean_m),
        deployed_covariance_sha256=_array_sha256(hybrid.covariance_m2),
        hybrid_artifact_id=hybrid_id,
        hybrid_reference_mean_identity_preserved=True,
        exact_physical_fallback_mean_identity_preserved=False,
        exact_physical_fallback_covariance_identity_preserved=False,
        metadata={} if metadata is None else metadata,
    )
    return ResidualHistoryDryRunResultV1(
        mean_m=hybrid.mean_m,
        covariance_m2=hybrid.covariance_m2,
        adapter=adapter,
        decision=decision,
        hybrid=hybrid,
    )


__all__ = [
    "ResidualHistoryDryRunDecisionV1",
    "ResidualHistoryDryRunResultV1",
    "run_source_only_residual_history_dry_run",
]
