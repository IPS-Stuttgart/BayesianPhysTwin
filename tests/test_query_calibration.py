import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.query_calibration import (
    QueryCalibrationV1,
    calibrate_query_covariance,
    fit_query_calibration,
    group_mahalanobis_nonconformity,
    load_query_calibration,
    query_group_is_covered,
    save_query_calibration,
)

_IDS = {
    "predictor_id": "1" * 64,
    "query_set_id": "2" * 64,
    "grouping_rule_id": "3" * 64,
    "guard_id": "4" * 64,
    "calibration_evidence_id": "5" * 64,
}


def _groups(count: int = 9) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    group_ids = [f"object-{index:02d}" for index in range(count)]
    residuals = [np.asarray([float(index + 1), 0.0]) for index in range(count)]
    covariances = [np.eye(2) for _ in range(count)]
    return group_ids, residuals, covariances


def _fit(*, covariance_scale: float = 1.0) -> QueryCalibrationV1:
    group_ids, residuals, covariances = _groups()
    return fit_query_calibration(
        group_ids,
        residuals,
        covariances,
        nominal_coverage=0.9,
        covariance_scale=covariance_scale,
        isotropic_variance=0.0,
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
        **_IDS,
    )


def test_fit_uses_one_maximum_score_per_independent_group() -> None:
    calibration = _fit()

    np.testing.assert_allclose(calibration.calibration_group_scores, np.arange(1, 10))
    assert calibration.finite_sample_rank == 9
    assert calibration.conformal_quantile == 9.0
    assert not calibration.calibration_group_scores.flags.writeable


def test_calibrated_covariance_and_group_coverage_share_one_scale() -> None:
    calibration = _fit()
    calibrated = calibrate_query_covariance(np.eye(2), calibration)

    np.testing.assert_allclose(calibrated, 81.0 * np.eye(2))
    assert not calibrated.flags.writeable
    assert query_group_is_covered(np.asarray([9.0, 0.0]), np.eye(2), calibration)
    assert not query_group_is_covered(
        np.asarray([9.01, 0.0]),
        np.eye(2),
        calibration,
    )


def test_group_score_is_invariant_to_row_order_and_duplicate_nonmaxima() -> None:
    residual = np.asarray([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    covariance = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    expected = group_mahalanobis_nonconformity(residual, covariance)

    permutation = np.asarray([2, 0, 1])
    reordered = group_mahalanobis_nonconformity(
        residual[permutation],
        covariance[permutation],
    )
    duplicated = group_mahalanobis_nonconformity(
        np.concatenate([residual, residual[[0]]], axis=0),
        np.concatenate([covariance, covariance[[0]]], axis=0),
    )

    assert expected == reordered == duplicated == 3.0


def test_affine_coordinate_change_preserves_mahalanobis_score() -> None:
    residual = np.asarray([[1.0, -2.0], [0.5, 1.5]])
    covariance = np.asarray(
        [
            [[2.0, 0.3], [0.3, 1.0]],
            [[1.5, -0.2], [-0.2, 0.8]],
        ]
    )
    transform = np.asarray([[2.0, 0.5], [-0.25, 1.25]])
    transformed_residual = residual @ transform.T
    transformed_covariance = np.asarray(
        [transform @ matrix @ transform.T for matrix in covariance]
    )

    original = group_mahalanobis_nonconformity(residual, covariance)
    transformed = group_mahalanobis_nonconformity(
        transformed_residual,
        transformed_covariance,
    )

    assert transformed == pytest.approx(original)


def test_covariance_scale_is_compensated_by_conformal_quantile() -> None:
    unit = _fit(covariance_scale=1.0)
    quadrupled = _fit(covariance_scale=4.0)

    assert quadrupled.conformal_quantile == pytest.approx(
        0.5 * unit.conformal_quantile
    )
    np.testing.assert_allclose(
        calibrate_query_covariance(np.eye(2), unit),
        calibrate_query_covariance(np.eye(2), quadrupled),
    )
    assert unit.artifact_id != quadrupled.artifact_id


def test_group_order_is_canonical_and_content_invariant() -> None:
    group_ids, residuals, covariances = _groups()
    forward = _fit()
    reverse = fit_query_calibration(
        list(reversed(group_ids)),
        list(reversed(residuals)),
        list(reversed(covariances)),
        nominal_coverage=0.9,
        covariance_scale=1.0,
        isotropic_variance=0.0,
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
        **_IDS,
    )

    assert forward.calibration_group_ids == tuple(sorted(group_ids))
    assert reverse.artifact_id == forward.artifact_id


class _UnreadableSequence(Sequence[np.ndarray]):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> np.ndarray:
        raise AssertionError(f"calibration outcome {index} was accessed")


def test_impossible_coverage_fails_before_outcomes_are_inspected() -> None:
    with pytest.raises(ValueError, match="infinite quantile"):
        fit_query_calibration(
            [f"object-{index}" for index in range(10)],
            _UnreadableSequence(),
            _UnreadableSequence(),
            nominal_coverage=0.95,
            covariance_scale=1.0,
            isotropic_variance=0.0,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
            **_IDS,
        )


def test_policy_selection_with_calibration_outcomes_is_rejected() -> None:
    group_ids, residuals, covariances = _groups()
    with pytest.raises(ValueError, match="cannot also select"):
        fit_query_calibration(
            group_ids,
            residuals,
            covariances,
            nominal_coverage=0.9,
            covariance_scale=1.0,
            isotropic_variance=0.0,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=True,
            **_IDS,
        )


def test_save_load_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    calibration = _fit()
    path = tmp_path / "query_calibration.json"
    save_query_calibration(calibration, path)
    loaded = load_query_calibration(path)

    assert loaded.artifact_id == calibration.artifact_id
    assert loaded.as_dict() == calibration.as_dict()

    record = json.loads(path.read_text(encoding="utf-8"))
    record["conformal_quantile"] = 8.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="conformal_quantile|artifact_id"):
        load_query_calibration(path)


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_query_calibration(path)


@pytest.mark.parametrize(
    ("residual", "covariance", "message"),
    [
        (np.asarray([1.0, 2.0]), np.eye(3), "shape"),
        (np.asarray([1.0, np.nan]), np.eye(2), "residual must be finite"),
        (
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 1.0], [0.0, 1.0]]),
            "symmetric",
        ),
        (
            np.asarray([1.0, 2.0]),
            np.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "positive definite",
        ),
    ],
)
def test_invalid_query_geometry_fails_closed(
    residual: np.ndarray,
    covariance: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        group_mahalanobis_nonconformity(residual, covariance)


def test_contract_rejects_changed_evidence_or_derived_quantile() -> None:
    calibration = _fit()
    values = calibration.as_dict()
    values.pop("artifact_id")
    values.pop("schema")
    values.pop("schema_version")
    values.pop("score")
    groups = values.pop("calibration_groups")
    group_ids = tuple(group["group_id"] for group in groups)
    scores = np.asarray([group["score"] for group in groups])

    wrong_quantile = dict(values)
    wrong_quantile["conformal_quantile"] = (
        calibration.conformal_quantile - 1.0
    )
    with pytest.raises(ValueError, match="conformal_quantile"):
        QueryCalibrationV1(
            calibration_group_ids=group_ids,
            calibration_group_scores=scores,
            **wrong_quantile,
        )

    changed = QueryCalibrationV1(
        calibration_group_ids=group_ids,
        calibration_group_scores=scores,
        calibration_evidence_id="6" * 64,
        **{
            key: value
            for key, value in values.items()
            if key != "calibration_evidence_id"
        },
    )
    assert changed.artifact_id != calibration.artifact_id


def test_duplicate_group_ids_and_noninteger_schema_versions_fail_closed(
    tmp_path: Path,
) -> None:
    group_ids, residuals, covariances = _groups()
    group_ids[-1] = group_ids[0]
    with pytest.raises(ValueError, match="unique"):
        fit_query_calibration(
            group_ids,
            residuals,
            covariances,
            nominal_coverage=0.9,
            covariance_scale=1.0,
            isotropic_variance=0.0,
            predictor_frozen_before_scores=True,
            calibration_outcomes_used_for_selection=False,
            **_IDS,
        )

    calibration = _fit()
    record = calibration.as_dict()
    record["schema_version"] = 1.0
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_query_calibration(path)
