"""Full-parameter covariance sources used by query portfolios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from ._canonical_contracts import plain_json
from ._portable_contracts import content_id
from ._posterior_covariance_portfolio_common import (
    POSTERIOR_COVARIANCE_SOURCE_SCHEMA,
    POSTERIOR_COVARIANCE_SOURCE_VERSION,
    FloatArray,
    array_record,
    portfolio_metadata,
    sha256_id,
    validated_covariance,
)
from .group_sandwich_covariance import GroupSandwichCovarianceResultV1
from .observed_information_covariance import ObservedInformationCovarianceResultV1
from .posterior_covariance_semantics import (
    PosteriorCovarianceMethod,
    PosteriorCovarianceSemanticsV1,
    exact_prior_fallback_covariance_semantics,
    working_irls_covariance_semantics,
)


@dataclass(frozen=True, slots=True)
class PosteriorCovarianceSourceV1:
    """One uncalibrated full-parameter covariance and its interpretation."""

    inference_result_id: str
    source_artifact_id: str
    covariance: FloatArray
    covariance_semantics: PosteriorCovarianceSemanticsV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        inference_id = sha256_id(
            self.inference_result_id,
            name="inference_result_id",
        )
        source_id = sha256_id(
            self.source_artifact_id,
            name="source_artifact_id",
        )
        covariance = validated_covariance(self.covariance, name="covariance")
        semantics = self.covariance_semantics
        if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
            raise ValueError(
                "covariance_semantics must be a PosteriorCovarianceSemanticsV1"
            )
        if semantics.calibrated:
            raise ValueError("portfolio sources must be uncalibrated")
        if semantics.dimension != len(covariance):
            raise ValueError("covariance_semantics dimension must match covariance")
        metadata = portfolio_metadata(
            self.metadata,
            name="posterior covariance source metadata",
        )

        object.__setattr__(self, "inference_result_id", inference_id)
        object.__setattr__(self, "source_artifact_id", source_id)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_id(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance source")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def method(self) -> PosteriorCovarianceMethod:
        return self.covariance_semantics.method

    @property
    def dimension(self) -> int:
        return len(self.covariance)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": POSTERIOR_COVARIANCE_SOURCE_SCHEMA,
            "schema_version": POSTERIOR_COVARIANCE_SOURCE_VERSION,
            "inference_result_id": self.inference_result_id,
            "source_artifact_id": self.source_artifact_id,
            "covariance": array_record(self.covariance),
            "covariance_semantics_id": self.covariance_semantics.artifact_id,
            "method": self.method,
            "dimension": self.dimension,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def working_covariance_source(
    inference_result_id: str,
    covariance: object,
    *,
    source_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSourceV1:
    """Bind one solver working covariance to a fixed inference identity."""

    matrix = validated_covariance(covariance, name="covariance")
    details = portfolio_metadata(
        metadata,
        name="working covariance source metadata",
        required={"portfolio_source": "working-irls"},
    )
    return PosteriorCovarianceSourceV1(
        inference_result_id=inference_result_id,
        source_artifact_id=source_artifact_id,
        covariance=matrix,
        covariance_semantics=working_irls_covariance_semantics(
            matrix,
            metadata=details,
        ),
        metadata=details,
    )


def observed_information_covariance_source(
    inference_result_id: str,
    result: ObservedInformationCovarianceResultV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSourceV1:
    """Adapt an observed-information result without changing its covariance."""

    if not isinstance(result, ObservedInformationCovarianceResultV1):
        raise TypeError("result must be an ObservedInformationCovarianceResultV1")
    details = portfolio_metadata(
        metadata,
        name="observed-information covariance source metadata",
        required={"adapter": "observed-information-v1"},
    )
    return PosteriorCovarianceSourceV1(
        inference_result_id=inference_result_id,
        source_artifact_id=cast(str, result.artifact_id),
        covariance=result.full_covariance,
        covariance_semantics=result.covariance_semantics,
        metadata=details,
    )


def group_sandwich_covariance_source(
    inference_result_id: str,
    result: GroupSandwichCovarianceResultV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSourceV1:
    """Adapt a group-sandwich result without changing its covariance."""

    if not isinstance(result, GroupSandwichCovarianceResultV1):
        raise TypeError("result must be a GroupSandwichCovarianceResultV1")
    details = portfolio_metadata(
        metadata,
        name="group-sandwich covariance source metadata",
        required={"adapter": "group-sandwich-v1"},
    )
    return PosteriorCovarianceSourceV1(
        inference_result_id=inference_result_id,
        source_artifact_id=cast(str, result.artifact_id),
        covariance=result.covariance,
        covariance_semantics=result.covariance_semantics,
        metadata=details,
    )


def exact_prior_fallback_covariance_source(
    inference_result_id: str,
    covariance: object,
    *,
    source_artifact_id: str,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSourceV1:
    """Bind the exact covariance retained after one rejected update."""

    matrix = validated_covariance(covariance, name="covariance")
    details = portfolio_metadata(
        metadata,
        name="fallback covariance source metadata",
        required={"portfolio_source": "exact-prior-fallback"},
    )
    return PosteriorCovarianceSourceV1(
        inference_result_id=inference_result_id,
        source_artifact_id=source_artifact_id,
        covariance=matrix,
        covariance_semantics=exact_prior_fallback_covariance_semantics(
            matrix,
            reason=reason,
            metadata=details,
        ),
        metadata=details,
    )


__all__ = [
    "PosteriorCovarianceSourceV1",
    "exact_prior_fallback_covariance_source",
    "group_sandwich_covariance_source",
    "observed_information_covariance_source",
    "working_covariance_source",
]
