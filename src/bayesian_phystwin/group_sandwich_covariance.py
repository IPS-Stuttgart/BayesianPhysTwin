"""Group-robust sandwich covariance for Bayesian-PhysTwin updates.

The estimator aggregates row scores exactly once within declared independent
correlation groups. It then combines the group score outer products with the
inverse local information matrix. The returned covariance is explicitly an
uncalibrated group-sandwich approximation; conformal calibration remains a
separate operation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import content_id
from .posterior_covariance_semantics import PosteriorCovarianceSemanticsV1

GROUP_SANDWICH_COVARIANCE_SCHEMA = "bayesian_phystwin.group_sandwich_covariance"
GROUP_SANDWICH_COVARIANCE_VERSION = 1

SmallSampleCorrection = Literal["none", "g_over_g_minus_one"]
SMALL_SAMPLE_CORRECTIONS: tuple[SmallSampleCorrection, ...] = (
    "none",
    "g_over_g_minus_one",
)

FloatArray = NDArray[np.float64]


def _literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _correction(value: object) -> SmallSampleCorrection:
    if type(value) is not str or value not in SMALL_SAMPLE_CORRECTIONS:
        raise ValueError(
            f"small_sample_correction must be one of {list(SMALL_SAMPLE_CORRECTIONS)}"
        )
    return cast(SmallSampleCorrection, value)


def _finite_matrix(value: object, *, name: str) -> FloatArray:
    try:
        matrix = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real matrix") from error
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _square_symmetric_matrix(value: object, *, name: str) -> FloatArray:
    matrix = _finite_matrix(value, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _positive_definite_bread(value: object) -> FloatArray:
    bread = _square_symmetric_matrix(value, name="bread")
    try:
        np.linalg.cholesky(bread)
    except np.linalg.LinAlgError as error:
        raise ValueError("bread must be positive definite") from error
    return bread


def _group_id_tuple(value: object, *, expected_count: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("group_ids must be a sequence of strings")
    group_ids = tuple(value)
    if len(group_ids) != expected_count:
        raise ValueError("group_ids length must equal score row count")
    return tuple(
        _literal_string(group_id, name=f"group_ids[{index}]")
        for index, group_id in enumerate(group_ids)
    )


def _finite_real(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _immutable(array: FloatArray) -> FloatArray:
    owned = np.array(array, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _array_record(array: FloatArray) -> dict[str, object]:
    canonical = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    return {
        "dtype": "float64-le",
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _project_numerical_psd(matrix: FloatArray) -> tuple[FloatArray, int]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-10 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ArithmeticError(
            "computed sandwich covariance is not positive semidefinite"
        )
    clipped = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * clipped) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    rank_tolerance = 1e-12 * max(1.0, float(np.max(clipped)))
    effective_rank = int(np.count_nonzero(clipped > rank_tolerance))
    return projected, effective_rank


@dataclass(frozen=True, slots=True)
class GroupSandwichCovarianceResultV1:
    """Content-addressed group-robust covariance result."""

    bread: FloatArray
    bread_inverse: FloatArray
    grouped_scores: FloatArray
    covariance: FloatArray
    group_ids: Sequence[str]
    group_row_counts: Sequence[int]
    small_sample_correction: SmallSampleCorrection
    correction_factor: float
    grouping_semantics: str
    minimum_group_count: int
    effective_rank: int
    covariance_semantics: PosteriorCovarianceSemanticsV1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        bread = _positive_definite_bread(self.bread)
        dimension = bread.shape[0]
        bread_inverse = _square_symmetric_matrix(
            self.bread_inverse,
            name="bread_inverse",
        )
        grouped_scores = _finite_matrix(
            self.grouped_scores,
            name="grouped_scores",
        )
        covariance = _square_symmetric_matrix(
            self.covariance,
            name="covariance",
        )
        if bread_inverse.shape != bread.shape:
            raise ValueError("bread_inverse shape must match bread")
        if grouped_scores.shape[1] != dimension:
            raise ValueError("grouped_scores width must match bread dimension")
        if covariance.shape != bread.shape:
            raise ValueError("covariance shape must match bread")
        identity = bread @ bread_inverse
        if not np.allclose(identity, np.eye(dimension), atol=1e-8, rtol=1e-8):
            raise ValueError("bread_inverse does not invert bread")

        group_ids = tuple(
            _literal_string(value, name=f"group_ids[{index}]")
            for index, value in enumerate(tuple(self.group_ids))
        )
        if len(group_ids) != grouped_scores.shape[0]:
            raise ValueError("group_ids length must match grouped_scores rows")
        if tuple(sorted(group_ids)) != group_ids or len(set(group_ids)) != len(
            group_ids
        ):
            raise ValueError("group_ids must be sorted and unique")

        row_counts = tuple(
            genuine_integer(
                value,
                name=f"group_row_counts[{index}]",
                minimum=1,
            )
            for index, value in enumerate(tuple(self.group_row_counts))
        )
        if len(row_counts) != len(group_ids):
            raise ValueError("group_row_counts length must match group_ids")
        if len(group_ids) < 2:
            raise ValueError("at least two groups are required")
        correction = _correction(self.small_sample_correction)
        correction_factor = _finite_real(
            self.correction_factor,
            name="correction_factor",
            minimum=1.0,
        )
        expected_factor = (
            1.0 if correction == "none" else len(group_ids) / (len(group_ids) - 1)
        )
        if not np.isclose(correction_factor, expected_factor, atol=0.0, rtol=1e-15):
            raise ValueError("correction_factor contradicts small_sample_correction")
        grouping_semantics = _literal_string(
            self.grouping_semantics,
            name="grouping_semantics",
        )
        minimum_group_count = genuine_integer(
            self.minimum_group_count,
            name="minimum_group_count",
            minimum=2,
        )
        if len(group_ids) < minimum_group_count:
            raise ValueError("group count is below minimum_group_count")
        effective_rank = genuine_integer(
            self.effective_rank,
            name="effective_rank",
            minimum=0,
        )
        if effective_rank > dimension:
            raise ValueError("effective_rank cannot exceed parameter dimension")
        expected_covariance, expected_rank = _project_numerical_psd(
            correction_factor
            * bread_inverse
            @ (grouped_scores.T @ grouped_scores)
            @ bread_inverse.T
        )
        if not np.allclose(
            covariance,
            expected_covariance,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(
                "covariance does not match bread, grouped scores, and correction"
            )
        if effective_rank != expected_rank:
            raise ValueError("effective_rank does not match covariance")
        covariance_eigenvalues = np.linalg.eigvalsh(covariance)
        covariance_tolerance = 1e-10 * max(
            1.0,
            float(np.max(np.abs(covariance_eigenvalues))),
        )
        if float(np.min(covariance_eigenvalues)) < -covariance_tolerance:
            raise ValueError("covariance must be positive semidefinite")

        semantics = self.covariance_semantics
        if not isinstance(semantics, PosteriorCovarianceSemanticsV1):
            raise ValueError(
                "covariance_semantics must be a PosteriorCovarianceSemanticsV1"
            )
        if semantics.method != "group_sandwich":
            raise ValueError("covariance_semantics method must be group_sandwich")
        if semantics.dimension != dimension:
            raise ValueError("covariance_semantics dimension must match bread")
        if not semantics.group_score_correction or semantics.mixture_curvature_exact:
            raise ValueError("covariance_semantics flags contradict group sandwich")
        if semantics.calibrated:
            raise ValueError("group sandwich covariance is not calibration by itself")
        if semantics.metadata.get("grouping_semantics") != grouping_semantics:
            raise ValueError("covariance_semantics does not bind grouping_semantics")
        if semantics.metadata.get("group_count") != len(group_ids):
            raise ValueError("covariance_semantics does not bind group_count")
        if semantics.metadata.get("small_sample_correction") != correction:
            raise ValueError(
                "covariance_semantics does not bind small_sample_correction"
            )

        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="group sandwich covariance metadata",
        )
        object.__setattr__(self, "bread", _immutable(bread))
        object.__setattr__(self, "bread_inverse", _immutable(bread_inverse))
        object.__setattr__(self, "grouped_scores", _immutable(grouped_scores))
        object.__setattr__(self, "covariance", _immutable(covariance))
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "group_row_counts", row_counts)
        object.__setattr__(self, "small_sample_correction", correction)
        object.__setattr__(self, "correction_factor", correction_factor)
        object.__setattr__(self, "grouping_semantics", grouping_semantics)
        object.__setattr__(self, "minimum_group_count", minimum_group_count)
        object.__setattr__(self, "effective_rank", effective_rank)
        object.__setattr__(self, "metadata", metadata)

        expected_id = content_id(self.descriptor())
        supplied_id = self.artifact_id
        if supplied_id is not None:
            supplied_id = literal_lower_hex(
                supplied_id,
                name="artifact_id",
                lengths={64},
            )
            if supplied_id != expected_id:
                raise ValueError("artifact_id does not match covariance content")
        object.__setattr__(self, "artifact_id", expected_id)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    @property
    def row_count(self) -> int:
        return sum(self.group_row_counts)

    @property
    def parameter_dimension(self) -> int:
        return self.bread.shape[0]

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": GROUP_SANDWICH_COVARIANCE_SCHEMA,
            "schema_version": GROUP_SANDWICH_COVARIANCE_VERSION,
            "bread": _array_record(self.bread),
            "bread_inverse": _array_record(self.bread_inverse),
            "grouped_scores": _array_record(self.grouped_scores),
            "covariance": _array_record(self.covariance),
            "group_count": self.group_count,
            "row_count": self.row_count,
            "parameter_dimension": self.parameter_dimension,
            "group_ids": list(self.group_ids),
            "group_row_counts": list(self.group_row_counts),
            "small_sample_correction": self.small_sample_correction,
            "correction_factor": self.correction_factor,
            "grouping_semantics": self.grouping_semantics,
            "minimum_group_count": self.minimum_group_count,
            "effective_rank": self.effective_rank,
            "covariance_semantics_id": self.covariance_semantics.artifact_id,
            "metadata": plain_json(self.metadata),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.descriptor(), "artifact_id": self.artifact_id}


def estimate_group_sandwich_covariance(
    bread: object,
    score_rows: object,
    group_ids: Sequence[str],
    *,
    grouping_semantics: str,
    likelihood_power_semantics: str = ("grouped-student-t-generalized-bayes-power-v1"),
    small_sample_correction: SmallSampleCorrection = "g_over_g_minus_one",
    minimum_group_count: int = 3,
    prior_included: bool = True,
    generalized_bayes: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> GroupSandwichCovarianceResultV1:
    """Estimate a covariance from declared independent group score sums.

    ``bread`` is the local positive-definite information matrix, including the
    prior curvature when ``prior_included`` is true. ``score_rows`` contains
    data-score contributions only. Rows sharing a ``group_id`` are summed before
    entering the meat matrix, so repeated rows inside one correlation group do
    not masquerade as additional independent evidence.
    """

    information = _positive_definite_bread(bread)
    scores = _finite_matrix(score_rows, name="score_rows")
    if scores.shape[1] != information.shape[0]:
        raise ValueError("score_rows width must match bread dimension")
    identifiers = _group_id_tuple(group_ids, expected_count=scores.shape[0])
    grouping = _literal_string(grouping_semantics, name="grouping_semantics")
    likelihood_semantics = _literal_string(
        likelihood_power_semantics,
        name="likelihood_power_semantics",
    )
    correction = _correction(small_sample_correction)
    minimum_groups = genuine_integer(
        minimum_group_count,
        name="minimum_group_count",
        minimum=2,
    )
    prior_flag = genuine_boolean(prior_included, name="prior_included")
    generalized_flag = genuine_boolean(
        generalized_bayes,
        name="generalized_bayes",
    )

    unique_group_ids = tuple(sorted(set(identifiers)))
    group_count = len(unique_group_ids)
    if group_count < minimum_groups:
        raise ValueError(
            f"at least {minimum_groups} independent groups are required; "
            f"received {group_count}"
        )
    group_index = {group_id: index for index, group_id in enumerate(unique_group_ids)}
    grouped_scores = np.zeros((group_count, information.shape[0]), dtype=np.float64)
    row_counts = np.zeros(group_count, dtype=np.int64)
    for score, group_id in zip(scores, identifiers, strict=True):
        index = group_index[group_id]
        grouped_scores[index] += score
        row_counts[index] += 1

    bread_inverse = np.linalg.solve(
        information,
        np.eye(information.shape[0], dtype=np.float64),
    )
    bread_inverse = 0.5 * (bread_inverse + bread_inverse.T)
    correction_factor = 1.0 if correction == "none" else group_count / (group_count - 1)
    meat = grouped_scores.T @ grouped_scores
    raw_covariance = correction_factor * bread_inverse @ meat @ bread_inverse.T
    covariance, effective_rank = _project_numerical_psd(raw_covariance)

    semantics = PosteriorCovarianceSemanticsV1(
        method="group_sandwich",
        dimension=information.shape[0],
        likelihood_power_semantics=likelihood_semantics,
        prior_included=prior_flag,
        generalized_bayes=generalized_flag,
        mixture_curvature_exact=False,
        group_score_correction=True,
        calibrated=False,
        metadata={
            "estimator_schema": GROUP_SANDWICH_COVARIANCE_SCHEMA,
            "estimator_schema_version": GROUP_SANDWICH_COVARIANCE_VERSION,
            "grouping_semantics": grouping,
            "group_count": group_count,
            "score_semantics": "data-score-rows-summed-once-per-declared-group",
            "small_sample_correction": correction,
        },
    )
    return GroupSandwichCovarianceResultV1(
        bread=information,
        bread_inverse=bread_inverse,
        grouped_scores=grouped_scores,
        covariance=covariance,
        group_ids=unique_group_ids,
        group_row_counts=tuple(int(value) for value in row_counts),
        small_sample_correction=correction,
        correction_factor=correction_factor,
        grouping_semantics=grouping,
        minimum_group_count=minimum_groups,
        effective_rank=effective_rank,
        covariance_semantics=semantics,
        metadata=metadata or {},
    )


__all__ = [
    "GROUP_SANDWICH_COVARIANCE_SCHEMA",
    "GROUP_SANDWICH_COVARIANCE_VERSION",
    "SMALL_SAMPLE_CORRECTIONS",
    "GroupSandwichCovarianceResultV1",
    "SmallSampleCorrection",
    "estimate_group_sandwich_covariance",
]
