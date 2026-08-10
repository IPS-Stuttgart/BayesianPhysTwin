"""Content-addressed common-query covariance portfolio contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    plain_json,
)
from ._portable_contracts import content_id
from ._posterior_covariance_portfolio_common import (
    ACCEPTED_METHODS,
    METHOD_ORDER,
    POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_SCHEMA,
    POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_VERSION,
    FloatArray,
    array_record,
    canonical_string,
    covariance_method,
    portfolio_metadata,
    projected_semantics,
    query_matrix_id,
    sha256_id,
    validated_query_matrix,
)
from ._posterior_covariance_sources import PosteriorCovarianceSourceV1
from .posterior_covariance_semantics import PosteriorCovarianceMethod
from .posterior_uncertainty import PosteriorQueryUncertaintyV1


@dataclass(frozen=True, slots=True)
class PosteriorQueryCovariancePortfolioV1:
    """Alternative raw query covariances with complete method accounting."""

    inference_result_id: str
    query_set_id: str
    query_matrix: FloatArray
    sources: Sequence[PosteriorCovarianceSourceV1]
    inference_admissible: bool
    reason: str
    unavailable_methods: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    entries: tuple[PosteriorQueryUncertaintyV1, ...] = field(
        init=False,
        repr=False,
    )
    _reference_method: PosteriorCovarianceMethod = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        inference_id = sha256_id(
            self.inference_result_id,
            name="inference_result_id",
        )
        query_set_id = sha256_id(self.query_set_id, name="query_set_id")
        query = validated_query_matrix(self.query_matrix)
        admitted = genuine_boolean(
            self.inference_admissible,
            name="inference_admissible",
        )
        reason = canonical_string(self.reason, name="reason")
        sources = self._validated_sources(inference_id, query.shape[1])
        methods = {source.method for source in sources}
        unavailable = self._validated_unavailable(methods)
        reference = self._validate_accounting(
            admitted=admitted,
            reason=reason,
            methods=methods,
            unavailable=set(unavailable),
        )
        if not admitted:
            fallback_reason = sources[0].covariance_semantics.metadata.get(
                "fallback_reason"
            )
            if fallback_reason != reason:
                raise ValueError(
                    "rejected portfolio reason must match fallback covariance reason"
                )
        query_digest = query_matrix_id(query)
        entries = tuple(
            self._entry(
                source,
                query=query,
                query_set_id=query_set_id,
                query_digest=query_digest,
            )
            for source in sources
        )
        metadata = portfolio_metadata(
            self.metadata,
            name="posterior covariance portfolio metadata",
        )

        object.__setattr__(self, "inference_result_id", inference_id)
        object.__setattr__(self, "query_set_id", query_set_id)
        object.__setattr__(self, "query_matrix", query)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "inference_admissible", admitted)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "unavailable_methods", unavailable)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_reference_method", reference)

        expected_id = content_id(self.descriptor())
        if self.artifact_id is not None:
            supplied_id = sha256_id(self.artifact_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match portfolio content")
        object.__setattr__(self, "artifact_id", expected_id)

    def _validated_sources(
        self,
        inference_id: str,
        query_width: int,
    ) -> tuple[PosteriorCovarianceSourceV1, ...]:
        if isinstance(self.sources, (str, bytes)) or not isinstance(
            self.sources,
            Sequence,
        ):
            raise ValueError("sources must be a sequence")
        sources = tuple(self.sources)
        if not sources or any(
            not isinstance(source, PosteriorCovarianceSourceV1)
            for source in sources
        ):
            raise ValueError(
                "sources must contain PosteriorCovarianceSourceV1 values"
            )
        if any(source.inference_result_id != inference_id for source in sources):
            raise ValueError("portfolio source inference_result_id changed")
        dimensions = {source.dimension for source in sources}
        if len(dimensions) != 1:
            raise ValueError("portfolio source dimensions changed")
        if dimensions.pop() != query_width:
            raise ValueError("query_matrix width must match covariance dimension")
        methods = tuple(source.method for source in sources)
        if len(set(methods)) != len(methods):
            raise ValueError("portfolio covariance source methods must be unique")
        return tuple(
            sorted(
                sources,
                key=lambda source: METHOD_ORDER[source.method],
            )
        )

    def _validated_unavailable(
        self,
        methods: set[PosteriorCovarianceMethod],
    ) -> Mapping[PosteriorCovarianceMethod, str]:
        if not isinstance(self.unavailable_methods, Mapping):
            raise ValueError("unavailable_methods must be a mapping")
        unavailable: dict[PosteriorCovarianceMethod, str] = {}
        for raw_method, raw_reason in self.unavailable_methods.items():
            method = covariance_method(
                raw_method,
                name="unavailable_methods key",
            )
            unavailable[method] = canonical_string(
                raw_reason,
                name=f"unavailable_methods[{method}]",
            )
        if methods & set(unavailable):
            raise ValueError(
                "a covariance method cannot be present and unavailable"
            )
        ordered = dict(
            sorted(
                unavailable.items(),
                key=lambda item: METHOD_ORDER[item[0]],
            )
        )
        return cast(
            Mapping[PosteriorCovarianceMethod, str],
            frozen_finite_json_mapping(
                ordered,
                name="posterior covariance unavailable methods",
            ),
        )

    @staticmethod
    def _validate_accounting(
        *,
        admitted: bool,
        reason: str,
        methods: set[PosteriorCovarianceMethod],
        unavailable: set[PosteriorCovarianceMethod],
    ) -> PosteriorCovarianceMethod:
        if admitted:
            if reason != "inference-admissible":
                raise ValueError(
                    "accepted portfolio reason must be inference-admissible"
                )
            if "irls_working" not in methods:
                raise ValueError(
                    "accepted portfolio requires irls_working covariance"
                )
            if "exact_prior_fallback" in methods or (
                "exact_prior_fallback" in unavailable
            ):
                raise ValueError(
                    "accepted portfolio cannot contain fallback covariance"
                )
            if methods | unavailable != set(ACCEPTED_METHODS):
                raise ValueError(
                    "accepted portfolio must contain or explain every raw method"
                )
            return "irls_working"
        if reason == "inference-admissible":
            raise ValueError("rejected portfolio must retain a rejection reason")
        if methods != {"exact_prior_fallback"}:
            raise ValueError(
                "rejected portfolio must contain only exact fallback"
            )
        if unavailable:
            raise ValueError(
                "rejected portfolio cannot list alternative methods"
            )
        return "exact_prior_fallback"

    @staticmethod
    def _entry(
        source: PosteriorCovarianceSourceV1,
        *,
        query: FloatArray,
        query_set_id: str,
        query_digest: str,
    ) -> PosteriorQueryUncertaintyV1:
        source_id = cast(str, source.artifact_id)
        projected = query @ source.covariance @ query.T
        projected = 0.5 * (projected + projected.T)
        semantics = projected_semantics(
            source.covariance_semantics,
            dimension=len(query),
            source_id=source_id,
            query_id=query_digest,
        )
        estimator_id = (
            None if source.method == "irls_working" else source.source_artifact_id
        )
        return PosteriorQueryUncertaintyV1(
            inference_result_id=source.inference_result_id,
            query_set_id=query_set_id,
            source_query_covariance_m2=projected,
            source_covariance_semantics=semantics,
            covariance_estimator_artifact_id=estimator_id,
            metadata={
                "posterior_covariance_source_id": source_id,
                "source_artifact_id": source.source_artifact_id,
                "query_matrix_sha256": query_digest,
                "parameter_dimension": source.dimension,
            },
        )

    @property
    def methods(self) -> tuple[PosteriorCovarianceMethod, ...]:
        return tuple(source.method for source in self.sources)

    @property
    def reference_method(self) -> PosteriorCovarianceMethod:
        return self._reference_method

    @property
    def query_matrix_sha256(self) -> str:
        return query_matrix_id(self.query_matrix)

    @property
    def parameter_dimension(self) -> int:
        return self.query_matrix.shape[1]

    @property
    def query_dimension(self) -> int:
        return self.query_matrix.shape[0]

    def entry(
        self,
        method: PosteriorCovarianceMethod,
    ) -> PosteriorQueryUncertaintyV1:
        requested = covariance_method(method, name="method")
        for entry in self.entries:
            if entry.source_covariance_semantics.method == requested:
                return entry
        raise KeyError(requested)

    def source(
        self,
        method: PosteriorCovarianceMethod,
    ) -> PosteriorCovarianceSourceV1:
        requested = covariance_method(method, name="method")
        for source in self.sources:
            if source.method == requested:
                return source
        raise KeyError(requested)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_SCHEMA,
            "schema_version": POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_VERSION,
            "inference_result_id": self.inference_result_id,
            "query_set_id": self.query_set_id,
            "query_matrix": array_record(self.query_matrix),
            "query_matrix_sha256": self.query_matrix_sha256,
            "parameter_dimension": self.parameter_dimension,
            "query_dimension": self.query_dimension,
            "methods": list(self.methods),
            "reference_method": self.reference_method,
            "inference_admissible": self.inference_admissible,
            "reason": self.reason,
            "source_artifact_ids": {
                source.method: source.artifact_id for source in self.sources
            },
            "entry_artifact_ids": {
                entry.source_covariance_semantics.method: entry.artifact_id
                for entry in self.entries
            },
            "unavailable_methods": dict(self.unavailable_methods),
            "selection_semantics": "no-implicit-covariance-winner-v1",
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.descriptor(),
            "sources": [source.to_record() for source in self.sources],
            "entries": [entry.to_record() for entry in self.entries],
            "artifact_id": self.artifact_id,
        }


def build_posterior_query_covariance_portfolio(
    inference_result_id: str,
    query_set_id: str,
    query_matrix: object,
    sources: Sequence[PosteriorCovarianceSourceV1],
    *,
    inference_admissible: bool,
    reason: str,
    unavailable_methods: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorQueryCovariancePortfolioV1:
    """Project every raw covariance source through one registered query."""

    return PosteriorQueryCovariancePortfolioV1(
        inference_result_id=inference_result_id,
        query_set_id=query_set_id,
        query_matrix=validated_query_matrix(query_matrix),
        sources=sources,
        inference_admissible=inference_admissible,
        reason=reason,
        unavailable_methods=(
            {} if unavailable_methods is None else unavailable_methods
        ),
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "PosteriorQueryCovariancePortfolioV1",
    "build_posterior_query_covariance_portfolio",
]
