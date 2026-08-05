from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from bayesian_phystwin.group_sandwich_covariance import (
    GroupSandwichCovarianceResultV1,
    estimate_group_sandwich_covariance,
)


def _example() -> GroupSandwichCovarianceResultV1:
    return estimate_group_sandwich_covariance(
        np.array([[2.0]]),
        np.array([[1.0], [2.0], [-1.0], [3.0]]),
        ["a", "a", "b", "c"],
        grouping_semantics="independent-object-v1",
    )


def test_matches_manual_group_cr1_covariance() -> None:
    result = _example()
    assert result.group_ids == ("a", "b", "c")
    assert result.group_row_counts == (2, 1, 1)
    np.testing.assert_allclose(result.grouped_scores, [[3.0], [-1.0], [3.0]])
    assert result.correction_factor == pytest.approx(1.5)
    np.testing.assert_allclose(result.covariance, [[7.125]])
    assert result.group_count == 3
    assert result.row_count == 4
    assert result.parameter_dimension == 1
    assert result.effective_rank == 1
    assert result.covariance_semantics.method == "group_sandwich"
    assert result.covariance_semantics.group_score_correction is True
    assert result.covariance_semantics.calibrated is False


def test_row_and_group_permutation_is_content_invariant() -> None:
    baseline = _example()
    permutation = np.array([3, 1, 2, 0])
    scores = np.array([[1.0], [2.0], [-1.0], [3.0]])[permutation]
    groups = np.array(["a", "a", "b", "c"])[permutation].tolist()
    permuted = estimate_group_sandwich_covariance(
        np.array([[2.0]]),
        scores,
        groups,
        grouping_semantics="independent-object-v1",
    )
    np.testing.assert_array_equal(permuted.grouped_scores, baseline.grouped_scores)
    np.testing.assert_array_equal(permuted.covariance, baseline.covariance)
    assert permuted.artifact_id == baseline.artifact_id


def test_duplicate_rows_stay_inside_one_group() -> None:
    baseline = _example()
    duplicated = estimate_group_sandwich_covariance(
        np.array([[2.0]]),
        np.array([[1.0], [2.0], [1.0], [-1.0], [3.0]]),
        ["a", "a", "a", "b", "c"],
        grouping_semantics="independent-object-v1",
    )
    assert duplicated.group_count == baseline.group_count
    assert duplicated.group_row_counts == (3, 1, 1)
    assert duplicated.correction_factor == baseline.correction_factor
    assert duplicated.covariance[0, 0] > baseline.covariance[0, 0]


def test_splitting_a_group_changes_the_claim_identity() -> None:
    baseline = _example()
    split = estimate_group_sandwich_covariance(
        np.array([[2.0]]),
        np.array([[1.0], [2.0], [-1.0], [3.0]]),
        ["a-1", "a-2", "b", "c"],
        grouping_semantics="artificial-row-split-diagnostic-v1",
    )
    assert split.group_count == 4
    assert split.artifact_id != baseline.artifact_id
    assert split.covariance_semantics.artifact_id != (
        baseline.covariance_semantics.artifact_id
    )


def test_arrays_are_immutable() -> None:
    result = _example()
    with pytest.raises(ValueError):
        result.covariance[0, 0] = 0.0


def test_result_rejects_covariance_not_generated_by_declared_scores() -> None:
    baseline = _example()
    with pytest.raises(ValueError, match="covariance does not match"):
        GroupSandwichCovarianceResultV1(
            bread=baseline.bread,
            bread_inverse=baseline.bread_inverse,
            grouped_scores=baseline.grouped_scores,
            covariance=np.zeros_like(baseline.covariance),
            group_ids=baseline.group_ids,
            group_row_counts=baseline.group_row_counts,
            small_sample_correction=baseline.small_sample_correction,
            correction_factor=baseline.correction_factor,
            grouping_semantics=baseline.grouping_semantics,
            minimum_group_count=baseline.minimum_group_count,
            effective_rank=0,
            covariance_semantics=baseline.covariance_semantics,
        )


def test_rejects_too_few_or_malformed_groups() -> None:
    with pytest.raises(ValueError, match="at least 3 independent groups"):
        estimate_group_sandwich_covariance(
            [[1.0]],
            [[1.0], [2.0]],
            ["only", "only"],
            grouping_semantics="session-v1",
        )
    with pytest.raises(ValueError, match="group_ids length"):
        estimate_group_sandwich_covariance(
            [[1.0]],
            [[1.0], [2.0]],
            ["a"],
            grouping_semantics="session-v1",
        )
    with pytest.raises(ValueError, match=r"group_ids\[0\]"):
        estimate_group_sandwich_covariance(
            [[1.0]],
            [[1.0], [2.0]],
            cast(list[str], [1, "b"]),
            grouping_semantics="session-v1",
        )


def test_rejects_invalid_bread_and_scores() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        estimate_group_sandwich_covariance(
            [[0.0]],
            [[1.0], [2.0], [3.0]],
            ["a", "b", "c"],
            grouping_semantics="session-v1",
        )
    with pytest.raises(ValueError, match="symmetric"):
        estimate_group_sandwich_covariance(
            [[2.0, 1.0], [0.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            ["a", "b", "c"],
            grouping_semantics="session-v1",
        )
    with pytest.raises(ValueError, match="finite"):
        estimate_group_sandwich_covariance(
            [[1.0]],
            [[np.nan], [2.0], [3.0]],
            ["a", "b", "c"],
            grouping_semantics="session-v1",
        )
