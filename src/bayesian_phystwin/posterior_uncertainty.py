"""Content-addressed uncertainty for registered physical queries.

BayesianPhysTwin exposes several covariance estimators and an independent-group
query calibration.  Their numerical arrays must not be mixed without retaining
which posterior result, query set, covariance interpretation, estimator
artifact, and calibration produced the reported uncertainty.  This module binds
those pieces without changing any point estimate or frozen solver.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_integer,
    immutable_array,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .calibration import (
    finite_group_conformal_rank,
    maximum_finite_group_coverage,
    minimum_groups_for_finite_conformal,
)
from .posterior_covariance_semantics import PosteriorCovarianceSemanticsV1
from .query_calibration import QueryCalibrationV1, calibrate_query_covariance

POSTERIOR_QUERY_UNCERTAINTY_SCHEMA = "bayesian_phystwin.posterior_query_uncertainty"
POSTERIOR_QUERY_UNCERTAINTY_VERSION = 1
FINITE_GROUP_COVERAGE_STATUS_SCHEMA = "bayesian_phystwin.finite_group_coverage_status"
FINITE_GROUP_COVERAGE_STATUS_VERSION = 1


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _array_record(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _validated_covariance(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    covariance = np.asarray(raw, dtype=np.float64)
    if covariance.ndim < 2 or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError(f"{name} must contain one or more square matrices")
    if covariance.size == 0 or covariance.shape[-1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        covariance,
        np.swapaxes(covariance, -1, -2),
        atol=1e-11,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")

    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    dimension = symmetric.shape[-1]
    eigenvalues = np.linalg.eigvalsh(symmetric.reshape((-1, dimension, dimension)))
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
    tolerance = 1e-12 + 1e-10 * scale
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")
    return immutable_array(symmetric, dtype=np.dtype("<f8"))


def _calibrated_semantics(
    source: PosteriorCovarianceSemanticsV1,
    calibration: QueryCalibrationV1,
) -> PosteriorCovarianceSemanticsV1:
    return PosteriorCovarianceSemanticsV1(
        method=source.method,
        dimension=source.dimension,
        likelihood_power_semantics=source.likelihood_power_semantics,
        prior_included=source.prior_included,
        generalized_bayes=source.generalized_bayes,
        mixture_curvature_exact=source.mixture_curvature_exact,
        group_score_correction=source.group_score_correction,
        calibrated=True,
        calibration_artifact_id=calibration.artifact_id,
        metadata=plain_json(source.metadata),
    )


@dataclass(frozen=True, slots=True)
class FiniteGroupCoverageStatusV1:
    """Target-blind feasibility of a finite split-conformal quantile."""

    independent_group_count: int
    requested_coverage: float
    finite_sample_rank: int = field(init=False)
    maximum_finite_coverage: float = field(init=False)
    minimum_required_group_count: int = field(init=False)
    finite: bool = field(init=False)
    _artifact_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        count = genuine_integer(
            self.independent_group_count,
            name="independent_group_count",
            minimum=1,
        )
        rank = finite_group_conformal_rank(count, self.requested_coverage)
        maximum = maximum_finite_group_coverage(count)
        minimum = minimum_groups_for_finite_conformal(self.requested_coverage)

        object.__setattr__(self, "independent_group_count", count)
        object.__setattr__(self, "requested_coverage", float(self.requested_coverage))
        object.__setattr__(self, "finite_sample_rank", rank)
        object.__setattr__(self, "maximum_finite_coverage", maximum)
        object.__setattr__(self, "minimum_required_group_count", minimum)
        object.__setattr__(self, "finite", rank <= count)
        object.__setattr__(self, "_artifact_id", content_id(self.descriptor()))

    @property
    def artifact_id(self) -> str:
        return self._artifact_id

    @property
    def status(self) -> str:
        return "finite" if self.finite else "unavailable"

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": FINITE_GROUP_COVERAGE_STATUS_SCHEMA,
            "schema_version": FINITE_GROUP_COVERAGE_STATUS_VERSION,
            "independent_group_count": self.independent_group_count,
            "requested_coverage": self.requested_coverage,
            "finite_sample_rank": self.finite_sample_rank,
            "maximum_finite_coverage": self.maximum_finite_coverage,
            "minimum_required_group_count": self.minimum_required_group_count,
            "finite": self.finite,
            "status": self.status,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class PosteriorQueryUncertaintyV1:
    """Bind posterior-derived query covariance to its interpretation and calibration.

    ``source_query_covariance_m2`` is the uncalibrated covariance in the exact
    registered query coordinates.  A supplied ``QueryCalibrationV1`` is applied
    only after its predictor and query-set identities match this artifact.
    """

    inference_result_id: str
    query_set_id: str
    source_query_covariance_m2: np.ndarray
    source_covariance_semantics: PosteriorCovarianceSemanticsV1
    covariance_estimator_artifact_id: str | None = None
    query_calibration: QueryCalibrationV1 | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None
    _predictor_id: str = field(init=False, repr=False, compare=False)
    _calibrated_query_covariance_m2: np.ndarray | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _reported_covariance_semantics: PosteriorCovarianceSemanticsV1 = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        inference_result_id = _sha256(
            self.inference_result_id,
            name="inference_result_id",
        )
        query_set_id = _sha256(self.query_set_id, name="query_set_id")
        covariance = _validated_covariance(
            self.source_query_covariance_m2,
            name="source_query_covariance_m2",
        )
        semantics = self.source_covariance_semantics
        if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
            raise ValueError(
                "source_covariance_semantics must be a PosteriorCovarianceSemanticsV1"
            )
        if semantics.calibrated:
            raise ValueError(
                "source_covariance_semantics must describe uncalibrated covariance"
            )
        if semantics.dimension != covariance.shape[-1]:
            raise ValueError(
                "source_covariance_semantics dimension must match query covariance"
            )

        estimator_id = self.covariance_estimator_artifact_id
        if estimator_id is not None:
            estimator_id = _sha256(
                estimator_id,
                name="covariance_estimator_artifact_id",
            )
        elif semantics.method != "irls_working":
            raise ValueError(
                "non-working covariance requires covariance_estimator_artifact_id"
            )
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="posterior query uncertainty metadata",
        )

        predictor_id = content_id(
            {
                "schema": "bayesian_phystwin.posterior_query_predictor",
                "schema_version": 1,
                "inference_result_id": inference_result_id,
                "query_set_id": query_set_id,
                "source_query_covariance_m2": _array_record(covariance),
                "source_covariance_semantics_id": semantics.artifact_id,
                "covariance_estimator_artifact_id": estimator_id,
            }
        )

        calibration = self.query_calibration
        calibrated_covariance: np.ndarray | None = None
        reported_semantics = semantics
        if calibration is not None:
            if not isinstance(calibration, QueryCalibrationV1):
                raise ValueError("query_calibration must be a QueryCalibrationV1")
            if calibration.predictor_id != predictor_id:
                raise ValueError(
                    "query_calibration predictor_id does not match source predictor"
                )
            if calibration.query_set_id != query_set_id:
                raise ValueError(
                    "query_calibration query_set_id does not match registered query"
                )
            calibrated_covariance = calibrate_query_covariance(
                covariance,
                calibration,
            )
            reported_semantics = _calibrated_semantics(semantics, calibration)

        object.__setattr__(self, "inference_result_id", inference_result_id)
        object.__setattr__(self, "query_set_id", query_set_id)
        object.__setattr__(self, "source_query_covariance_m2", covariance)
        object.__setattr__(
            self,
            "covariance_estimator_artifact_id",
            estimator_id,
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_predictor_id", predictor_id)
        object.__setattr__(
            self,
            "_calibrated_query_covariance_m2",
            calibrated_covariance,
        )
        object.__setattr__(
            self,
            "_reported_covariance_semantics",
            reported_semantics,
        )

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = _sha256(supplied_id, name="artifact_id")
            if supplied_id != expected_id:
                raise ValueError(
                    "artifact_id does not match posterior uncertainty content"
                )
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def predictor_id(self) -> str:
        """Identity calibrated before outcomes are used."""

        return self._predictor_id

    @property
    def calibrated(self) -> bool:
        return self.query_calibration is not None

    @property
    def calibrated_query_covariance_m2(self) -> np.ndarray | None:
        return self._calibrated_query_covariance_m2

    @property
    def reported_query_covariance_m2(self) -> np.ndarray:
        calibrated = self._calibrated_query_covariance_m2
        return self.source_query_covariance_m2 if calibrated is None else calibrated

    @property
    def reported_covariance_semantics(self) -> PosteriorCovarianceSemanticsV1:
        return self._reported_covariance_semantics

    @property
    def nominal_coverage(self) -> float | None:
        calibration = self.query_calibration
        return None if calibration is None else calibration.nominal_coverage

    @property
    def calibration_group_count(self) -> int | None:
        calibration = self.query_calibration
        return None if calibration is None else len(calibration.calibration_group_ids)

    def descriptor(self) -> dict[str, object]:
        calibrated = self._calibrated_query_covariance_m2
        calibration = self.query_calibration
        return {
            "schema": POSTERIOR_QUERY_UNCERTAINTY_SCHEMA,
            "schema_version": POSTERIOR_QUERY_UNCERTAINTY_VERSION,
            "inference_result_id": self.inference_result_id,
            "query_set_id": self.query_set_id,
            "predictor_id": self.predictor_id,
            "source_query_covariance_m2": _array_record(
                self.source_query_covariance_m2
            ),
            "source_covariance_semantics_id": (
                self.source_covariance_semantics.artifact_id
            ),
            "covariance_estimator_artifact_id": (self.covariance_estimator_artifact_id),
            "query_calibration_id": (
                None if calibration is None else calibration.artifact_id
            ),
            "calibrated_query_covariance_m2": (
                None if calibrated is None else _array_record(calibrated)
            ),
            "reported_covariance_semantics_id": (
                self.reported_covariance_semantics.artifact_id
            ),
            "calibrated": self.calibrated,
            "nominal_coverage": self.nominal_coverage,
            "calibration_group_count": self.calibration_group_count,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def finite_group_coverage_status(
    independent_group_count: int,
    requested_coverage: float,
) -> FiniteGroupCoverageStatusV1:
    """Return finite-group feasibility without reading calibration outcomes."""

    return FiniteGroupCoverageStatusV1(
        independent_group_count=independent_group_count,
        requested_coverage=requested_coverage,
    )


__all__ = [
    "FINITE_GROUP_COVERAGE_STATUS_SCHEMA",
    "FINITE_GROUP_COVERAGE_STATUS_VERSION",
    "POSTERIOR_QUERY_UNCERTAINTY_SCHEMA",
    "POSTERIOR_QUERY_UNCERTAINTY_VERSION",
    "FiniteGroupCoverageStatusV1",
    "PosteriorQueryUncertaintyV1",
    "finite_group_coverage_status",
]
