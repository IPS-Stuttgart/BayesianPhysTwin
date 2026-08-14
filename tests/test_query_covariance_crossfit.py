from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from bayesian_phystwin.query_covariance_crossfit import (
    QueryCovarianceCrossFitV1,
    StructuredQueryCovarianceCandidateV1,
    StructuredQueryCovarianceTransformV1,
    apply_structured_query_covariance,
    fit_cross_fitted_query_covariance,
    group_gaussian_energy_score,
    score_query_covariance_group,
)

DIGEST = "a" * 64


def _anisotropic_groups() -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    ids = ["object-c", "object-a", "object-b"]
    residual = np.array([[2.0, 0.0], [-2.0, 0.0]])
    covariance = np.repeat(np.eye(2)[None], 2, axis=0)
    return ids, [residual.copy() for _ in ids], [covariance.copy() for _ in ids]


def _fit(
    candidates: Sequence[StructuredQueryCovarianceCandidateV1],
    *,
    maximum_worst_group_regret: float = 10.0,
) -> QueryCovarianceCrossFitV1:
    ids, residuals, covariances = _anisotropic_groups()
    return fit_cross_fitted_query_covariance(
        ids,
        residuals,
        covariances,
        candidates,
        predictor_id=DIGEST,
        query_set_id="b" * 64,
        grouping_rule_id="c" * 64,
        development_evidence_id="d" * 64,
        reference_candidate_id=candidates[0].candidate_id,
        maximum_worst_group_regret=maximum_worst_group_regret,
        hyperparameter_grid_frozen_before_scores=True,
        target_outcomes_used=False,
    )


def test_cross_fit_selects_rank_one_excess_and_is_permutation_invariant() -> None:
    raw = StructuredQueryCovarianceCandidateV1()
    rank_one = StructuredQueryCovarianceCandidateV1(
        low_rank_rank=1,
        low_rank_fraction=1.0,
    )
    result = _fit([raw, rank_one])
    assert result.selected_candidate_id == rank_one.candidate_id
    assert result.selected_mean_group_nll < np.mean(
        result.cross_validated_group_nll[:, result.reference_index]
    )
    assert result.selected_transform.numerical_low_rank == 1
    np.testing.assert_allclose(
        result.selected_transform.low_rank_covariance,
        np.diag([3.0, 0.0]),
        atol=1e-12,
    )

    ids, residuals, covariances = _anisotropic_groups()
    reverse = fit_cross_fitted_query_covariance(
        ids[::-1],
        residuals[::-1],
        covariances[::-1],
        [raw, rank_one],
        predictor_id=DIGEST,
        query_set_id="b" * 64,
        grouping_rule_id="c" * 64,
        development_evidence_id="d" * 64,
        reference_candidate_id=raw.candidate_id,
        maximum_worst_group_regret=10.0,
        hyperparameter_grid_frozen_before_scores=True,
        target_outcomes_used=False,
    )
    assert reverse.artifact_id == result.artifact_id


def test_worst_group_regret_guard_retains_reference() -> None:
    raw = StructuredQueryCovarianceCandidateV1()
    inflated = StructuredQueryCovarianceCandidateV1(covariance_scale=4.0)
    group_ids = ["high-a", "high-b", "high-c", "zero"]
    high = np.array([[2.0, 0.0], [-2.0, 0.0]])
    zero = np.zeros((2, 2))
    covariances = [np.repeat(np.eye(2)[None], 2, axis=0) for _ in group_ids]
    result = fit_cross_fitted_query_covariance(
        group_ids,
        [high, high, high, zero],
        covariances,
        [raw, inflated],
        predictor_id=DIGEST,
        query_set_id="b" * 64,
        grouping_rule_id="c" * 64,
        development_evidence_id="d" * 64,
        reference_candidate_id=raw.candidate_id,
        maximum_worst_group_regret=0.0,
        hyperparameter_grid_frozen_before_scores=True,
        target_outcomes_used=False,
    )
    assert result.selected_candidate_id == raw.candidate_id
    assert result.selected_worst_group_regret == 0.0


def test_transform_is_psd_immutable_and_preserves_cross_axis_structure() -> None:
    candidate = StructuredQueryCovarianceCandidateV1(
        covariance_scale=2.0,
        diagonal_shrinkage=0.25,
        isotropic_variance=0.5,
        low_rank_rank=1,
        low_rank_fraction=0.5,
    )
    transform = StructuredQueryCovarianceTransformV1(
        candidate=candidate,
        dimension=2,
        low_rank_covariance=np.diag([1.0, 0.0]),
        source_group_ids=("object-a", "object-b"),
        source_evidence_id=DIGEST,
    )
    covariance = np.array([[2.0, 1.0], [1.0, 3.0]])
    result = apply_structured_query_covariance(covariance, transform)
    expected = 2.0 * np.array([[2.0, 0.75], [0.75, 3.0]])
    expected += 0.5 * np.eye(2) + np.diag([1.0, 0.0])
    np.testing.assert_allclose(result, expected)
    assert np.min(np.linalg.eigvalsh(result)) > 0.0
    assert not result.flags.writeable
    try:
        result.setflags(write=True)
    except ValueError:
        pass
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result[0, 0] = -1.0


def test_diagnostics_and_energy_score_are_finite_and_group_balanced() -> None:
    result = _fit(
        [
            StructuredQueryCovarianceCandidateV1(),
            StructuredQueryCovarianceCandidateV1(
                low_rank_rank=1,
                low_rank_fraction=1.0,
            ),
        ]
    )
    residual = np.array([[2.0, 0.0], [-2.0, 0.0]])
    covariance = np.repeat(np.eye(2)[None], 2, axis=0)
    diagnostics = score_query_covariance_group(
        residual,
        covariance,
        result.selected_transform,
        squared_ellipsoid_radius=1.1,
    )
    assert diagnostics.endpoint_count == 2
    assert diagnostics.mean_mahalanobis_squared == pytest.approx(1.0)
    assert diagnostics.ellipsoid_coverage == 1.0
    assert diagnostics.mean_effective_rank > 1.0
    assert diagnostics.maximum_condition_number == pytest.approx(4.0)

    samples = np.array(
        [
            [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
            [[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]],
        ]
    )
    score = group_gaussian_energy_score(
        residual,
        covariance,
        result.selected_transform,
        standard_normal_sample_pairs=samples,
    )
    assert np.isfinite(score)
    assert score > 0.0


class _BombSequence(Sequence[object]):
    def __len__(self) -> int:
        raise AssertionError("outcomes were inspected")

    def __getitem__(self, index: int) -> object:
        raise AssertionError("outcomes were inspected")

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("outcomes were inspected")


def test_information_order_fails_before_outcomes_are_inspected() -> None:
    with pytest.raises(ValueError, match="grid must be frozen"):
        fit_cross_fitted_query_covariance(
            ["a", "b"],
            _BombSequence(),
            _BombSequence(),
            [StructuredQueryCovarianceCandidateV1()],
            predictor_id=DIGEST,
            query_set_id="b" * 64,
            grouping_rule_id="c" * 64,
            development_evidence_id="d" * 64,
            reference_candidate_id=(
                StructuredQueryCovarianceCandidateV1().candidate_id
            ),
            hyperparameter_grid_frozen_before_scores=False,
            target_outcomes_used=False,
        )
    with pytest.raises(ValueError, match="target outcomes"):
        fit_cross_fitted_query_covariance(
            ["a", "b"],
            _BombSequence(),
            _BombSequence(),
            [StructuredQueryCovarianceCandidateV1()],
            predictor_id=DIGEST,
            query_set_id="b" * 64,
            grouping_rule_id="c" * 64,
            development_evidence_id="d" * 64,
            reference_candidate_id=(
                StructuredQueryCovarianceCandidateV1().candidate_id
            ),
            hyperparameter_grid_frozen_before_scores=True,
            target_outcomes_used=True,
        )


def test_invalid_candidates_and_singular_scoring_fail_closed() -> None:
    with pytest.raises(ValueError, match="fraction must be zero"):
        StructuredQueryCovarianceCandidateV1(low_rank_fraction=0.5)
    with pytest.raises(ValueError, match="fraction must be positive"):
        StructuredQueryCovarianceCandidateV1(low_rank_rank=1)

    raw = StructuredQueryCovarianceCandidateV1()
    with pytest.raises(ValueError, match="unique"):
        _fit([raw, raw])

    transform = StructuredQueryCovarianceTransformV1(
        candidate=raw,
        dimension=2,
        low_rank_covariance=np.zeros((2, 2)),
        source_group_ids=("a", "b"),
        source_evidence_id=DIGEST,
    )
    with pytest.raises(ValueError, match="positive definite"):
        score_query_covariance_group(
            np.ones(2),
            np.zeros((2, 2)),
            transform,
            squared_ellipsoid_radius=1.0,
        )


def test_artifact_replays_selection_and_detects_tampering() -> None:
    result = _fit(
        [
            StructuredQueryCovarianceCandidateV1(),
            StructuredQueryCovarianceCandidateV1(
                low_rank_rank=1,
                low_rank_fraction=1.0,
            ),
        ]
    )
    scores = np.array(result.cross_validated_group_nll, copy=True)
    scores[:, result.reference_index] -= 100.0
    with pytest.raises(ValueError, match="does not replay"):
        QueryCovarianceCrossFitV1(
            predictor_id=result.predictor_id,
            query_set_id=result.query_set_id,
            grouping_rule_id=result.grouping_rule_id,
            development_evidence_id=result.development_evidence_id,
            development_group_ids=result.development_group_ids,
            candidates=result.candidates,
            cross_validated_group_nll=scores,
            reference_candidate_id=result.reference_candidate_id,
            selected_candidate_id=result.selected_candidate_id,
            maximum_worst_group_regret=result.maximum_worst_group_regret,
            selected_transform=result.selected_transform,
            hyperparameter_grid_frozen_before_scores=True,
            target_outcomes_used=False,
        )


def test_explicit_reference_makes_candidate_order_canonical() -> None:
    raw = StructuredQueryCovarianceCandidateV1()
    rank_one = StructuredQueryCovarianceCandidateV1(
        low_rank_rank=1,
        low_rank_fraction=1.0,
    )
    ids, residuals, covariances = _anisotropic_groups()
    forward = fit_cross_fitted_query_covariance(
        ids,
        residuals,
        covariances,
        [raw, rank_one],
        predictor_id=DIGEST,
        query_set_id="b" * 64,
        grouping_rule_id="c" * 64,
        development_evidence_id="d" * 64,
        reference_candidate_id=raw.candidate_id,
        maximum_worst_group_regret=10.0,
        hyperparameter_grid_frozen_before_scores=True,
        target_outcomes_used=False,
    )
    reverse = fit_cross_fitted_query_covariance(
        ids,
        residuals,
        covariances,
        [rank_one, raw],
        predictor_id=DIGEST,
        query_set_id="b" * 64,
        grouping_rule_id="c" * 64,
        development_evidence_id="d" * 64,
        reference_candidate_id=raw.candidate_id,
        maximum_worst_group_regret=10.0,
        hyperparameter_grid_frozen_before_scores=True,
        target_outcomes_used=False,
    )
    assert reverse.artifact_id == forward.artifact_id
