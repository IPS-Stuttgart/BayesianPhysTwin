"""Group-cross-fitted selection for structured physical-query covariance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._canonical_contracts import canonical_string_tuple
from ._query_covariance_crossfit_common import (
    QUERY_COVARIANCE_CROSSFIT_CLAIM_BOUNDARY,
    QUERY_COVARIANCE_CROSSFIT_SCHEMA,
    QUERY_COVARIANCE_CROSSFIT_SCORE,
    QUERY_COVARIANCE_CROSSFIT_VERSION,
    StructuredQueryCovarianceCandidateV1,
    StructuredQueryCovarianceTransformV1,
    _canonical_groups,
    _cholesky,
    _content_id,
    _finite_real,
    _fit_transform,
    _immutable_array,
    _numeric_array,
    _sha256,
    apply_structured_query_covariance,
)


def _selection_index(
    scores: np.ndarray,
    candidate_ids: tuple[str, ...],
    *,
    reference_index: int,
    maximum_worst_group_regret: float,
) -> int:
    reference = scores[:, reference_index]
    eligible: list[int] = []
    for index in range(scores.shape[1]):
        worst_regret = float(np.max(scores[:, index] - reference, initial=-np.inf))
        if worst_regret <= maximum_worst_group_regret + 1e-12:
            eligible.append(index)
    if reference_index not in eligible:  # pragma: no cover - exact zero regret.
        raise AssertionError("reference candidate must remain eligible")
    return min(
        eligible,
        key=lambda index: (
            float(np.mean(scores[:, index])),
            float(np.max(scores[:, index], initial=-np.inf)),
            float(np.median(scores[:, index])),
            candidate_ids[index],
        ),
    )


@dataclass(frozen=True, slots=True)
class QueryCovarianceCrossFitV1:
    """Content-addressed group-cross-fitted covariance-selection result."""

    predictor_id: str
    query_set_id: str
    grouping_rule_id: str
    development_evidence_id: str
    development_group_ids: tuple[str, ...]
    candidates: tuple[StructuredQueryCovarianceCandidateV1, ...]
    cross_validated_group_nll: np.ndarray
    reference_candidate_id: str
    selected_candidate_id: str
    maximum_worst_group_regret: float
    selected_transform: StructuredQueryCovarianceTransformV1
    hyperparameter_grid_frozen_before_scores: bool
    target_outcomes_used: bool

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "query_set_id",
            "grouping_rule_id",
            "development_evidence_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        group_ids = canonical_string_tuple(
            self.development_group_ids,
            name="development_group_ids",
            allow_empty=False,
        )
        if (
            tuple(sorted(group_ids)) != group_ids
            or len(set(group_ids)) != len(group_ids)
        ):
            raise ValueError("development_group_ids must be sorted and unique")
        if len(group_ids) < 2:
            raise ValueError("at least two development groups are required")
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be a nonempty tuple")
        if not all(
            isinstance(candidate, StructuredQueryCovarianceCandidateV1)
            for candidate in self.candidates
        ):
            raise TypeError(
                "every candidate must be StructuredQueryCovarianceCandidateV1"
            )
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate transforms must be unique")
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise ValueError("candidates must be sorted by candidate_id")
        scores = _numeric_array(
            self.cross_validated_group_nll,
            name="cross_validated_group_nll",
        )
        if scores.shape != (len(group_ids), len(candidate_ids)):
            raise ValueError("cross_validated_group_nll shape changed")
        reference_id = _sha256(
            self.reference_candidate_id,
            name="reference_candidate_id",
        )
        selected_id = _sha256(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        if reference_id not in candidate_ids or selected_id not in candidate_ids:
            raise ValueError("reference and selected candidate IDs must be present")
        threshold = _finite_real(
            self.maximum_worst_group_regret,
            name="maximum_worst_group_regret",
            minimum=0.0,
        )
        if type(self.hyperparameter_grid_frozen_before_scores) is not bool:
            raise ValueError(
                "hyperparameter_grid_frozen_before_scores must be a boolean"
            )
        if type(self.target_outcomes_used) is not bool:
            raise ValueError("target_outcomes_used must be a boolean")
        if not self.hyperparameter_grid_frozen_before_scores:
            raise ValueError("the hyperparameter grid must be frozen before scoring")
        if self.target_outcomes_used:
            raise ValueError("target outcomes may not be used for source selection")
        expected_index = _selection_index(
            scores,
            candidate_ids,
            reference_index=candidate_ids.index(reference_id),
            maximum_worst_group_regret=threshold,
        )
        if candidate_ids[expected_index] != selected_id:
            raise ValueError("selected_candidate_id does not replay the selection rule")
        if not isinstance(
            self.selected_transform,
            StructuredQueryCovarianceTransformV1,
        ):
            raise TypeError(
                "selected_transform must be StructuredQueryCovarianceTransformV1"
            )
        if self.selected_transform.candidate.candidate_id != selected_id:
            raise ValueError("selected_transform does not match selected_candidate_id")
        if self.selected_transform.source_group_ids != group_ids:
            raise ValueError("selected_transform source groups changed")
        if self.selected_transform.source_evidence_id != self.development_evidence_id:
            raise ValueError("selected_transform source evidence changed")
        object.__setattr__(self, "development_group_ids", group_ids)
        object.__setattr__(
            self,
            "cross_validated_group_nll",
            _immutable_array(scores, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "reference_candidate_id", reference_id)
        object.__setattr__(self, "selected_candidate_id", selected_id)
        object.__setattr__(self, "maximum_worst_group_regret", threshold)

    @property
    def selected_index(self) -> int:
        return tuple(candidate.candidate_id for candidate in self.candidates).index(
            self.selected_candidate_id
        )

    @property
    def reference_index(self) -> int:
        return tuple(candidate.candidate_id for candidate in self.candidates).index(
            self.reference_candidate_id
        )

    @property
    def selected_mean_group_nll(self) -> float:
        return float(np.mean(self.cross_validated_group_nll[:, self.selected_index]))

    @property
    def selected_worst_group_nll(self) -> float:
        return float(np.max(self.cross_validated_group_nll[:, self.selected_index]))

    @property
    def selected_worst_group_regret(self) -> float:
        selected = self.cross_validated_group_nll[:, self.selected_index]
        reference = self.cross_validated_group_nll[:, self.reference_index]
        return float(np.max(selected - reference))

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": QUERY_COVARIANCE_CROSSFIT_SCHEMA,
            "schema_version": QUERY_COVARIANCE_CROSSFIT_VERSION,
            "selection_score": QUERY_COVARIANCE_CROSSFIT_SCORE,
            "claim_boundary": QUERY_COVARIANCE_CROSSFIT_CLAIM_BOUNDARY,
            "predictor_id": self.predictor_id,
            "query_set_id": self.query_set_id,
            "grouping_rule_id": self.grouping_rule_id,
            "development_evidence_id": self.development_evidence_id,
            "development_group_ids": list(self.development_group_ids),
            "candidates": [candidate.descriptor() for candidate in self.candidates],
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "cross_validated_group_nll": self.cross_validated_group_nll.tolist(),
            "reference_candidate_id": self.reference_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "maximum_worst_group_regret": self.maximum_worst_group_regret,
            "selected_transform": self.selected_transform.descriptor(),
            "selected_transform_id": self.selected_transform.transform_id,
            "hyperparameter_grid_frozen_before_scores": (
                self.hyperparameter_grid_frozen_before_scores
            ),
            "target_outcomes_used": self.target_outcomes_used,
        }

    @property
    def artifact_id(self) -> str:
        return _content_id(self.descriptor())

    def as_dict(self) -> dict[str, Any]:
        result = self.descriptor()
        result["artifact_id"] = self.artifact_id
        return result


def _group_gaussian_nll(
    residual: np.ndarray,
    covariance: np.ndarray,
    transform: StructuredQueryCovarianceTransformV1,
) -> float:
    transformed = apply_structured_query_covariance(covariance, transform)
    total = 0.0
    dimension = residual.shape[1]
    constant = dimension * np.log(2.0 * np.pi)
    for index, (error, matrix) in enumerate(
        zip(residual, transformed, strict=True)
    ):
        factor = _cholesky(matrix, name=f"transformed covariance {index}")
        whitened = np.linalg.solve(factor, error)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
        total += 0.5 * (constant + log_determinant + float(whitened @ whitened))
    value = total / len(residual)
    if not np.isfinite(value):
        raise ValueError("group Gaussian NLL must be finite")
    return float(value)


def fit_cross_fitted_query_covariance(
    development_group_ids: Sequence[str],
    residual_groups: Sequence[object],
    covariance_groups: Sequence[object],
    candidates: Sequence[StructuredQueryCovarianceCandidateV1],
    *,
    predictor_id: str,
    query_set_id: str,
    grouping_rule_id: str,
    development_evidence_id: str,
    reference_candidate_id: str,
    maximum_worst_group_regret: float = 0.0,
    hyperparameter_grid_frozen_before_scores: bool,
    target_outcomes_used: bool,
) -> QueryCovarianceCrossFitV1:
    """Select a structured covariance transform by leave-one-group-out NLL.

    Information-order declarations and candidate identities are checked before
    any residual or covariance element is inspected.
    """

    for name, value in (
        ("predictor_id", predictor_id),
        ("query_set_id", query_set_id),
        ("grouping_rule_id", grouping_rule_id),
        ("development_evidence_id", development_evidence_id),
    ):
        _sha256(value, name=name)
    if type(hyperparameter_grid_frozen_before_scores) is not bool:
        raise ValueError("hyperparameter_grid_frozen_before_scores must be a boolean")
    if type(target_outcomes_used) is not bool:
        raise ValueError("target_outcomes_used must be a boolean")
    if not hyperparameter_grid_frozen_before_scores:
        raise ValueError("the hyperparameter grid must be frozen before scoring")
    if target_outcomes_used:
        raise ValueError("target outcomes may not be used for source selection")
    threshold = _finite_real(
        maximum_worst_group_regret,
        name="maximum_worst_group_regret",
        minimum=0.0,
    )
    input_candidates = tuple(candidates)
    if not input_candidates:
        raise ValueError("candidates must not be empty")
    if not all(
        isinstance(candidate, StructuredQueryCovarianceCandidateV1)
        for candidate in input_candidates
    ):
        raise TypeError("every candidate must be StructuredQueryCovarianceCandidateV1")
    input_candidate_ids = tuple(
        candidate.candidate_id for candidate in input_candidates
    )
    if len(set(input_candidate_ids)) != len(input_candidate_ids):
        raise ValueError("candidate transforms must be unique")
    reference_id = _sha256(
        reference_candidate_id,
        name="reference_candidate_id",
    )
    candidate_tuple = tuple(
        sorted(input_candidates, key=lambda candidate: candidate.candidate_id)
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in candidate_tuple)
    if reference_id not in candidate_ids:
        raise ValueError("reference_candidate_id must identify one candidate")

    group_ids, residuals, covariances, dimension = _canonical_groups(
        development_group_ids,
        residual_groups,
        covariance_groups,
    )
    if any(candidate.low_rank_rank > dimension for candidate in candidate_tuple):
        raise ValueError("candidate low_rank_rank exceeds the query dimension")

    scores = np.empty((len(group_ids), len(candidate_tuple)), dtype=np.float64)
    for held_out in range(len(group_ids)):
        training_ids = tuple(
            group_id
            for index, group_id in enumerate(group_ids)
            if index != held_out
        )
        training_residuals = tuple(
            residual
            for index, residual in enumerate(residuals)
            if index != held_out
        )
        training_covariances = tuple(
            covariance
            for index, covariance in enumerate(covariances)
            if index != held_out
        )
        for candidate_index, candidate in enumerate(candidate_tuple):
            transform = _fit_transform(
                training_ids,
                training_residuals,
                training_covariances,
                candidate,
                source_evidence_id=development_evidence_id,
            )
            scores[held_out, candidate_index] = _group_gaussian_nll(
                residuals[held_out],
                covariances[held_out],
                transform,
            )

    selected_index = _selection_index(
        scores,
        candidate_ids,
        reference_index=candidate_ids.index(reference_id),
        maximum_worst_group_regret=threshold,
    )
    selected_transform = _fit_transform(
        group_ids,
        residuals,
        covariances,
        candidate_tuple[selected_index],
        source_evidence_id=development_evidence_id,
    )
    return QueryCovarianceCrossFitV1(
        predictor_id=predictor_id,
        query_set_id=query_set_id,
        grouping_rule_id=grouping_rule_id,
        development_evidence_id=development_evidence_id,
        development_group_ids=group_ids,
        candidates=candidate_tuple,
        cross_validated_group_nll=scores,
        reference_candidate_id=reference_id,
        selected_candidate_id=candidate_ids[selected_index],
        maximum_worst_group_regret=threshold,
        selected_transform=selected_transform,
        hyperparameter_grid_frozen_before_scores=(
            hyperparameter_grid_frozen_before_scores
        ),
        target_outcomes_used=target_outcomes_used,
    )


__all__ = [
    "QueryCovarianceCrossFitV1",
    "fit_cross_fitted_query_covariance",
]
