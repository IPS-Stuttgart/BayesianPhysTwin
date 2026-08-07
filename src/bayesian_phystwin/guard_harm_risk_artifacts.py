"""Artifact-bound harmful accepted-update risk certification.

The lower-level finite-group certificate accepts a verified exact-fallback mask.
This module supplies the claim-bearing boundary: it derives that mask from the
selected and fallback artifact identities, binds the derivation into a separate
content address, and refuses to certify a rejected group whose selected artifact
is not exactly the registered fallback artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import content_id, sha256_digest
from .guard_harm_risk import (
    GuardHarmRiskCertificateV1,
    certify_guard_harm_risk,
)

GUARD_FALLBACK_ARTIFACT_BINDING_SCHEMA = (
    "bayesian_phystwin.guard_fallback_artifact_binding"
)
GUARD_FALLBACK_ARTIFACT_BINDING_VERSION = 1
GUARD_HARM_RISK_ARTIFACT_CERTIFICATE_SCHEMA = (
    "bayesian_phystwin.guard_harm_risk_artifact_certificate"
)
GUARD_HARM_RISK_ARTIFACT_CERTIFICATE_VERSION = 1

BoolArray = NDArray[np.bool_]


def _canonical_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _canonical_groups(value: Sequence[str]) -> tuple[str, ...]:
    groups = tuple(
        _canonical_string(item, name=f"group_ids[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if not groups:
        raise ValueError("group_ids must not be empty")
    if len(set(groups)) != len(groups):
        raise ValueError("group_ids must not contain duplicates")
    return groups


def _artifact_ids(
    value: Sequence[str],
    *,
    name: str,
    expected_count: int,
) -> tuple[str, ...]:
    identities = tuple(
        sha256_digest(item, name=f"{name}[{index}]")
        for index, item in enumerate(tuple(value))
    )
    if len(identities) != expected_count:
        raise ValueError(f"{name} length must match group_ids")
    return identities


def _immutable_bool(value: BoolArray) -> BoolArray:
    canonical = np.asarray(value, dtype=np.bool_, order="C")
    return np.frombuffer(
        canonical.tobytes(order="C"),
        dtype=np.bool_,
    ).reshape(canonical.shape)


@dataclass(frozen=True, slots=True)
class GuardFallbackArtifactBindingV1:
    """Content-bound selected/fallback equality for independent groups."""

    group_ids: Sequence[str]
    selected_artifact_ids: Sequence[str]
    fallback_artifact_ids: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        groups = _canonical_groups(self.group_ids)
        selected = _artifact_ids(
            self.selected_artifact_ids,
            name="selected_artifact_ids",
            expected_count=len(groups),
        )
        fallback = _artifact_ids(
            self.fallback_artifact_ids,
            name="fallback_artifact_ids",
            expected_count=len(groups),
        )
        order = tuple(sorted(range(len(groups)), key=groups.__getitem__))
        groups = tuple(groups[index] for index in order)
        selected = tuple(selected[index] for index in order)
        fallback = tuple(fallback[index] for index in order)
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="guard fallback artifact binding metadata",
        )
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "selected_artifact_ids", selected)
        object.__setattr__(self, "fallback_artifact_ids", fallback)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match fallback binding")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def exact_fallback_mask(self) -> BoolArray:
        values = np.asarray(
            [
                selected == fallback
                for selected, fallback in zip(
                    self.selected_artifact_ids,
                    self.fallback_artifact_ids,
                    strict=True,
                )
            ],
            dtype=np.bool_,
        )
        return _immutable_bool(values)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GUARD_FALLBACK_ARTIFACT_BINDING_SCHEMA,
            "schema_version": GUARD_FALLBACK_ARTIFACT_BINDING_VERSION,
            "group_ids": list(self.group_ids),
            "selected_artifact_ids": list(self.selected_artifact_ids),
            "fallback_artifact_ids": list(self.fallback_artifact_ids),
            "exact_fallback_mask": self.exact_fallback_mask.tolist(),
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class GuardHarmRiskArtifactCertificateV1:
    """Risk certificate bound to recomputed selected/fallback equality."""

    fallback_binding: GuardFallbackArtifactBindingV1
    risk_certificate: GuardHarmRiskCertificateV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        binding = self.fallback_binding
        certificate = self.risk_certificate
        if not isinstance(binding, GuardFallbackArtifactBindingV1):
            raise TypeError("fallback_binding must be a GuardFallbackArtifactBindingV1")
        if not isinstance(certificate, GuardHarmRiskCertificateV1):
            raise TypeError("risk_certificate must be a GuardHarmRiskCertificateV1")
        if tuple(certificate.group_ids) != tuple(binding.group_ids):
            raise ValueError(
                "risk-certificate groups differ from fallback-binding groups"
            )
        if not np.array_equal(
            certificate.fallback_identity_verified,
            binding.exact_fallback_mask,
        ):
            raise ValueError(
                "risk-certificate fallback verification differs from artifact IDs"
            )
        rejected = ~np.asarray(certificate.accepted_mask, dtype=np.bool_)
        if np.any(~binding.exact_fallback_mask[rejected]):
            raise ValueError(
                "every rejected group must select the exact fallback artifact"
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="guard harm-risk artifact certificate metadata",
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = sha256_digest(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError(
                    "artifact_id does not match artifact-bound certificate"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def certified(self) -> bool:
        return self.risk_certificate.certified

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GUARD_HARM_RISK_ARTIFACT_CERTIFICATE_SCHEMA,
            "schema_version": (GUARD_HARM_RISK_ARTIFACT_CERTIFICATE_VERSION),
            "fallback_binding_id": self.fallback_binding.artifact_id,
            "risk_certificate_id": self.risk_certificate.artifact_id,
            "certified": self.certified,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def certify_guard_harm_risk_from_artifacts(
    *,
    guard_policy_id: str,
    threshold_source_artifact_id: str,
    certification_partition_id: str,
    statistical_unit: str,
    metric: str,
    threshold_selection_group_ids: Sequence[str],
    group_ids: Sequence[str],
    risk_scores: object,
    candidate_losses: object,
    fallback_losses: object,
    selected_artifact_ids: Sequence[str],
    fallback_artifact_ids: Sequence[str],
    threshold: float,
    harm_margin: float,
    target_harm_probability: float,
    confidence_level: float,
    minimum_accepted_group_count: int,
    threshold_frozen_before_certification_outcomes: bool,
    certification_outcomes_used_for_threshold_selection: bool,
    certification_groups_independent: bool,
    binding_metadata: Mapping[str, Any] | None = None,
    certificate_metadata: Mapping[str, Any] | None = None,
) -> GuardHarmRiskArtifactCertificateV1:
    """Certify one frozen guard while recomputing exact fallback equality."""

    binding = GuardFallbackArtifactBindingV1(
        group_ids=group_ids,
        selected_artifact_ids=selected_artifact_ids,
        fallback_artifact_ids=fallback_artifact_ids,
        metadata={} if binding_metadata is None else binding_metadata,
    )
    indexed = {group: index for index, group in enumerate(group_ids)}
    order = tuple(indexed[group] for group in binding.group_ids)
    scores = np.asarray(risk_scores)[list(order)]
    candidate = np.asarray(candidate_losses)[list(order)]
    fallback = np.asarray(fallback_losses)[list(order)]
    certificate = certify_guard_harm_risk(
        guard_policy_id=guard_policy_id,
        threshold_source_artifact_id=threshold_source_artifact_id,
        certification_partition_id=certification_partition_id,
        statistical_unit=statistical_unit,
        metric=metric,
        threshold_selection_group_ids=threshold_selection_group_ids,
        group_ids=binding.group_ids,
        risk_scores=scores,
        candidate_losses=candidate,
        fallback_losses=fallback,
        fallback_identity_verified=binding.exact_fallback_mask,
        threshold=threshold,
        harm_margin=harm_margin,
        target_harm_probability=target_harm_probability,
        confidence_level=confidence_level,
        minimum_accepted_group_count=minimum_accepted_group_count,
        threshold_frozen_before_certification_outcomes=(
            threshold_frozen_before_certification_outcomes
        ),
        certification_outcomes_used_for_threshold_selection=(
            certification_outcomes_used_for_threshold_selection
        ),
        certification_groups_independent=certification_groups_independent,
        metadata={} if certificate_metadata is None else certificate_metadata,
    )
    return GuardHarmRiskArtifactCertificateV1(
        fallback_binding=binding,
        risk_certificate=certificate,
        metadata={
            "fallback_equality_source": ("selected-and-fallback-content-identities-v1")
        },
    )


__all__ = [
    "GUARD_FALLBACK_ARTIFACT_BINDING_SCHEMA",
    "GUARD_FALLBACK_ARTIFACT_BINDING_VERSION",
    "GUARD_HARM_RISK_ARTIFACT_CERTIFICATE_SCHEMA",
    "GUARD_HARM_RISK_ARTIFACT_CERTIFICATE_VERSION",
    "GuardFallbackArtifactBindingV1",
    "GuardHarmRiskArtifactCertificateV1",
    "certify_guard_harm_risk_from_artifacts",
]
