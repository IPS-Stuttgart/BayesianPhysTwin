"""Auditable portfolios of alternative posterior query covariances.

A portfolio keeps working IRLS, exact observed-information, and group-sandwich
covariances attached to one inference result and one registered query. It never
selects a preferred covariance automatically and never treats an uncalibrated
portfolio member as a coverage guarantee.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .group_sandwich_covariance import GroupSandwichCovarianceResultV1
from .observed_information_covariance import ObservedInformationCovarianceResultV1
from .posterior_covariance_semantics import (
    POSTERIOR_COVARIANCE_METHODS,
    PosteriorCovarianceMethod,
    PosteriorCovarianceSemanticsV1,
    exact_prior_fallback_covariance_semantics,
    working_irls_covariance_semantics,
)
from .posterior_uncertainty import PosteriorQueryUncertaintyV1

POSTERIOR_COVARIANCE_SOURCE_SCHEMA = (
    "bayesian_phystwin.posterior_covariance_source"
)
POSTERIOR_COVARIANCE_SOURCE_VERSION = 1
POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_SCHEMA = (
    "bayesian_phystwin.posterior_query_covariance_portfolio"
)
POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_VERSION = 1

_ACCEPTED_METHODS: tuple[PosteriorCovarianceMethod, ...] = (
    "irls_working",
    "laplace_observed_information",
    "group_sandwich",
)
_METHOD_ORDER: dict[PosteriorCovarianceMethod, int] = {
    method: index for index, method in enumerate(POSTERIOR_COVARIANCE_METHODS)
}

FloatArray: TypeAlias = NDArray[np.float64]


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _method(value: object, *, name: str) -> PosteriorCovarianceMethod:
    if type(value) is not str or value not in POSTERIOR_COVARIANCE_METHODS:
        raise ValueError(
            f"{name} must be one of {list(POSTERIOR_COVARIANCE_METHODS)}"
        )
    return cast(PosteriorCovarianceMethod, value)


def _validated_covariance(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    covariance = np.asarray(raw, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not len(covariance) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be nonempty and finite")
    if not np.allclose(covariance, covariance.T, atol=1e-11, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-12 + 1e-10 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return immutable_array(symmetric, dtype=np.dtype("<f8"))


def _validated_query_matrix(value: object) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("query_matrix must contain real numeric values")
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("query_matrix must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("query_matrix must be finite")
    return immutable_array(matrix, dtype=np.dtype("<f8"))


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _query_matrix_sha256(query_matrix: FloatArray) -> str:
    return content_id(
        {
            "schema": "bayesian_phystwin.posterior_covariance_query_matrix",
            "schema_version": 1,
            "query_matrix": _array_record(query_matrix),
        }
    )


def _metadata_with_required(
    metadata: Mapping[str, Any] | None,
    *,
    required: Mapping[str, object],
) -> dict[str, Any]:
    details = {} if metadata is None else dict(metadata)
    for key, value in required.items():
        existing = details.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"metadata contradicts {key}")
        details[key] = value
    return details


def _projected_semantics(
    source: PosteriorCovarianceSemanticsV1,
    *,
    dimension: int,
    source_id: str,
    query_matrix_sha256: str,
) -> PosteriorCovarianceSemanticsV1:
    details = dict(plain_json(source.metadata))
    required = {
        "source_covariance_semantics_id": source.artifact_id,
        "source_covariance_source_id": source_id,
        "query_matrix_sha256": query_matrix_sha256,
    }
    for key, value in required.items():
        existing = details.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"source covariance metadata contradicts {key}")
        details[key] = value
    return PosteriorCovarianceSemanticsV1(
        method=source.method,
        dimension=dimension,
        likelihood_power_semantics=source.likelihood_power_semantics,
        prior_included=source.prior_included,
        generalized_bayes=source.generalized_bayes,
        mixture_curvature_exact=source.mixture_curvature_exact,
        group_score_correction=source.group_score_correction,
        calibrated=False,
        metadata=details,
    )


@dataclass(frozen=True, slots=True)
class PosteriorCovarianceSourceV1:
    """One uncalibrated full-parameter covariance for a fixed inference result."""

    inference_result_id: str
    source_artifact_id: str
    covariance: FloatArray
    covariance_semantics: PosteriorCovarianceSemanticsV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        inference_result_id = _sha256(
            self.inference_result_id,
            name="inference_result_id",
        )
        source_artifact_id = _sha256(
            self.source_artifact_id,
            name="source_artifact_id",
        )
        covariance = _validated_covariance(self.covariance, name="covariance")
        semantics = self.covariance_semantics
        if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
            raise ValueError(
                "covariance_semantics must be a PosteriorCovarianceSemanticsV1"
            )
        if semantics.calibrated:
            raise ValueError("portfolio sources must be uncalibrated")
        if semantics.dimension != len(covariance):
            raise ValueError("covariance_semantics dimension must match covariance")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="posterior covariance source metadata",
        )
        object.__setattr__(self, "inference_result_id", inference_result_id)
        object.__setattr__(self, "source_artifact_id", source_artifact_id)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance source content")
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
            "covariance": _array_record(self.covariance),
            "covariance_semantics_id": self.covariance_semantics.artifact_id,
            "method": self.method,
            "dimension": self.dimension,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class PosteriorQueryCovariancePortfolioV1:
    """Alternative raw query covariances with no implicit winner selection."""

    inference_result_id: str
    query_set_id: str
    query_matrix: FloatArray
    sources: Sequence[PosteriorCovarianceSourceV1]
    entries: Sequence[PosteriorQueryUncertaintyV1]
    reference_method: PosteriorCovarianceMethod
    inference_admissible: bool
    reason: str
    unavailable_methods: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        inference_result_id = _sha256(
            self.inference_result_id,
            name="inference_result_id",
        )
        query_set_id = _sha256(self.query_set_id, name="query_set_id")
        query = _validated_query_matrix(self.query_matrix)
        query_matrix_sha256 = _query_matrix_sha256(query)

        if isinstance(self.sources, (str, bytes)) or not isinstance(
            self.sources,
            Sequence,
        ):
            raise ValueError("sources must be a sequence")
        sources = tuple(self.sources)
        if not sources:
            raise ValueError("sources must be nonempty")
        if any(not isinstance(value, PosteriorCovarianceSourceV1) for value in sources):
            raise ValueError("sources must contain PosteriorCovarianceSourceV1 values")
        if any(value.inference_result_id != inference_result_id for value in sources):
            raise ValueError("portfolio source inference_result_id changed")
        dimensions = {value.dimension for value in sources}
        if len(dimensions) != 1:
            raise ValueError("portfolio source dimensions changed")
        parameter_dimension = dimensions.pop()
        if query.shape[1] != parameter_dimension:
            raise ValueError("query_matrix width must match covariance dimension")
        source_methods = tuple(value.method for value in sources)
        if len(set(source_methods)) != len(source_methods):
            raise ValueError("portfolio covariance source methods must be unique")
        ordered_sources = tuple(
            sorted(sources, key=lambda value: _METHOD_ORDER[value.method])
        )

        if isinstance(self.entries, (str, bytes)) or not isinstance(
            self.entries,
            Sequence,
        ):
            raise ValueError("entries must be a sequence")
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("entries must be nonempty")
        for entry in entries:
            if not isinstance(entry, PosteriorQueryUncertaintyV1):
                raise ValueError(
                    "entries must contain PosteriorQueryUncertaintyV1 values"
                )
            if entry.query_calibration is not None:
                raise ValueError("portfolio entries must remain uncalibrated")
            if entry.inference_result_id != inference_result_id:
                raise ValueError("portfolio entry inference_result_id changed")
            if entry.query_set_id != query_set_id:
                raise ValueError("portfolio entry query_set_id changed")
            if entry.source_query_covariance_m2.ndim != 2:
                raise ValueError("portfolio entries require one query covariance")
        entry_methods = tuple(
            entry.source_covariance_semantics.method for entry in entries
        )
        if len(set(entry_methods)) != len(entry_methods):
            raise ValueError("portfolio covariance entry methods must be unique")
        if set(entry_methods) != set(source_methods):
            raise ValueError("portfolio sources and entries cover different methods")
        ordered_entries = tuple(
            sorted(
                entries,
                key=lambda entry: _METHOD_ORDER[
                    entry.source_covariance_semantics.method
                ],
            )
        )
        query_dimension = query.shape[0]
        source_by_method = {value.method: value for value in ordered_sources}
        for entry in ordered_entries:
            method = entry.source_covariance_semantics.method
            source = source_by_method[method]
            source_id = cast(str, source.artifact_id)
            expected_covariance = query @ source.covariance @ query.T
            expected_covariance = 0.5 * (
                expected_covariance + expected_covariance.T
            )
            if not np.allclose(
                entry.source_query_covariance_m2,
                expected_covariance,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError("portfolio query covariance does not match source")
            if entry.source_query_covariance_m2.shape != (
                query_dimension,
                query_dimension,
            ):
                raise ValueError("portfolio query covariance dimensions changed")
            entry_metadata = entry.metadata
            if entry_metadata.get("posterior_covariance_source_id") != source_id:
                raise ValueError("portfolio entry source identity changed")
            if entry_metadata.get("source_artifact_id") != source.source_artifact_id:
                raise ValueError("portfolio entry estimator identity changed")
            if entry_metadata.get("query_matrix_sha256") != query_matrix_sha256:
                raise ValueError("portfolio entry query matrix identity changed")
            if entry_metadata.get("parameter_dimension") != parameter_dimension:
                raise ValueError("portfolio entry parameter dimension changed")
            semantics = entry.source_covariance_semantics
            semantics_metadata = semantics.metadata
            if semantics_metadata.get("query_matrix_sha256") != query_matrix_sha256:
                raise ValueError(
                    "portfolio covariance semantics query identity changed"
                )
            if semantics_metadata.get("source_covariance_source_id") != source_id:
                raise ValueError("portfolio covariance source identity changed")
            if semantics_metadata.get("source_covariance_semantics_id") != (
                source.covariance_semantics.artifact_id
            ):
                raise ValueError("portfolio source covariance semantics changed")
            if method == "irls_working":
                if entry.covariance_estimator_artifact_id is not None:
                    raise ValueError("working covariance must not name an estimator")
            elif entry.covariance_estimator_artifact_id != source.source_artifact_id:
                raise ValueError("portfolio covariance estimator identity changed")

        reference_method = _method(
            self.reference_method,
            name="reference_method",
        )
        method_set = set(source_methods)
        if reference_method not in method_set:
            raise ValueError("reference_method is not present in sources")
        inference_admissible = genuine_boolean(
            self.inference_admissible,
            name="inference_admissible",
        )
        reason = _literal_string(self.reason, name="reason")
        if not isinstance(self.unavailable_methods, Mapping):
            raise ValueError("unavailable_methods must be a mapping")
        unavailable: dict[PosteriorCovarianceMethod, str] = {}
        for raw_method, raw_reason in self.unavailable_methods.items():
            method = _method(raw_method, name="unavailable_methods key")
            unavailable[method] = _literal_string(
                raw_reason,
                name=f"unavailable_methods[{method}]",
            )
        if method_set & set(unavailable):
            raise ValueError("a covariance method cannot be present and unavailable")

        if inference_admissible:
            if reason != "inference-admissible":
                raise ValueError(
                    "accepted portfolio reason must be inference-admissible"
                )
            if reference_method != "irls_working":
                raise ValueError("accepted portfolio reference must be irls_working")
            if "irls_working" not in method_set:
                raise ValueError("accepted portfolio requires irls_working covariance")
            if "exact_prior_fallback" in method_set or (
                "exact_prior_fallback" in unavailable
            ):
                raise ValueError(
                    "accepted portfolio cannot contain fallback covariance"
                )
            accounted = method_set | set(unavailable)
            if accounted != set(_ACCEPTED_METHODS):
                raise ValueError(
                    "accepted portfolio must contain or explain every raw method"
                )
        else:
            if reason == "inference-admissible":
                raise ValueError("rejected portfolio must retain a rejection reason")
            if reference_method != "exact_prior_fallback":
                raise ValueError("rejected portfolio reference must be exact fallback")
            if method_set != {"exact_prior_fallback"}:
                raise ValueError("rejected portfolio must contain only exact fallback")
            if unavailable:
                raise ValueError("rejected portfolio cannot list alternative methods")

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="posterior query covariance portfolio metadata",
        )
        object.__setattr__(self, "inference_result_id", inference_result_id)
        object.__setattr__(self, "query_set_id", query_set_id)
        object.__setattr__(self, "query_matrix", query)
        object.__setattr__(self, "sources", ordered_sources)
        object.__setattr__(self, "entries", ordered_entries)
        object.__setattr__(self, "reference_method", reference_method)
        object.__setattr__(self, "inference_admissible", inference_admissible)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "unavailable_methods",
            dict(sorted(unavailable.items(), key=lambda item: _METHOD_ORDER[item[0]])),
        )
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match portfolio content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def methods(self) -> tuple[PosteriorCovarianceMethod, ...]:
        return tuple(value.method for value in self.sources)

    @property
    def query_matrix_sha256(self) -> str:
        return _query_matrix_sha256(self.query_matrix)

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
        requested = _method(method, name="method")
        for entry in self.entries:
            if entry.source_covariance_semantics.method == requested:
                return entry
        raise KeyError(requested)

    def source(
        self,
        method: PosteriorCovarianceMethod,
    ) -> PosteriorCovarianceSourceV1:
        requested = _method(method, name="method")
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
            "query_matrix": _array_record(self.query_matrix),
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


def working_covariance_source(
    inference_result_id: str,
    covariance: object,
    *,
    source_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> PosteriorCovarianceSourceV1:
    """Bind one solver working covariance to a fixed inference identity."""

    matrix = _validated_covariance(covariance, name="covariance")
    details = _metadata_with_required(
        metadata,
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
    """Adapt an observed-information artifact without changing its covariance."""

    if not isinstance(result, ObservedInformationCovarianceResultV1):
        raise TypeError("result must be an ObservedInformationCovarianceResultV1")
    details = _metadata_with_required(
        metadata,
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
    """Adapt a group-sandwich artifact without changing its covariance."""

    if not isinstance(result, GroupSandwichCovarianceResultV1):
        raise TypeError("result must be a GroupSandwichCovarianceResultV1")
    details = _metadata_with_required(
        metadata,
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

    matrix = _validated_covariance(covariance, name="covariance")
    details = _metadata_with_required(
        metadata,
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
    """Project raw covariance alternatives through one registered linear query."""

    result_id = _sha256(inference_result_id, name="inference_result_id")
    registered_query_id = _sha256(query_set_id, name="query_set_id")
    query = _validated_query_matrix(query_matrix)
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise ValueError("sources must be a sequence")
    covariance_sources = tuple(sources)
    if not covariance_sources:
        raise ValueError("sources must be nonempty")
    if any(
        not isinstance(source, PosteriorCovarianceSourceV1)
        for source in covariance_sources
    ):
        raise ValueError("sources must contain PosteriorCovarianceSourceV1 values")
    if any(source.inference_result_id != result_id for source in covariance_sources):
        raise ValueError("source inference_result_id changed")
    dimensions = {source.dimension for source in covariance_sources}
    if len(dimensions) != 1:
        raise ValueError("source covariance dimensions changed")
    parameter_dimension = dimensions.pop()
    if query.shape[1] != parameter_dimension:
        raise ValueError("query_matrix width must match covariance dimension")
    methods = [source.method for source in covariance_sources]
    if len(set(methods)) != len(methods):
        raise ValueError("source covariance methods must be unique")

    query_matrix_sha256 = _query_matrix_sha256(query)
    entries: list[PosteriorQueryUncertaintyV1] = []
    for source in covariance_sources:
        source_id = cast(str, source.artifact_id)
        projected = query @ source.covariance @ query.T
        query_covariance = 0.5 * (projected + projected.T)
        semantics = _projected_semantics(
            source.covariance_semantics,
            dimension=len(query),
            source_id=source_id,
            query_matrix_sha256=query_matrix_sha256,
        )
        estimator_id = (
            None if source.method == "irls_working" else source.source_artifact_id
        )
        entries.append(
            PosteriorQueryUncertaintyV1(
                inference_result_id=result_id,
                query_set_id=registered_query_id,
                source_query_covariance_m2=query_covariance,
                source_covariance_semantics=semantics,
                covariance_estimator_artifact_id=estimator_id,
                metadata={
                    "posterior_covariance_source_id": source_id,
                    "source_artifact_id": source.source_artifact_id,
                    "query_matrix_sha256": query_matrix_sha256,
                    "parameter_dimension": parameter_dimension,
                },
            )
        )

    admitted = genuine_boolean(
        inference_admissible,
        name="inference_admissible",
    )
    reference: PosteriorCovarianceMethod = (
        "irls_working" if admitted else "exact_prior_fallback"
    )
    return PosteriorQueryCovariancePortfolioV1(
        inference_result_id=result_id,
        query_set_id=registered_query_id,
        query_matrix=query,
        sources=covariance_sources,
        entries=entries,
        reference_method=reference,
        inference_admissible=admitted,
        reason=reason,
        unavailable_methods={} if unavailable_methods is None else unavailable_methods,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "POSTERIOR_COVARIANCE_SOURCE_SCHEMA",
    "POSTERIOR_COVARIANCE_SOURCE_VERSION",
    "POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_SCHEMA",
    "POSTERIOR_QUERY_COVARIANCE_PORTFOLIO_VERSION",
    "PosteriorCovarianceSourceV1",
    "PosteriorQueryCovariancePortfolioV1",
    "build_posterior_query_covariance_portfolio",
    "exact_prior_fallback_covariance_source",
    "group_sandwich_covariance_source",
    "observed_information_covariance_source",
    "working_covariance_source",
]
