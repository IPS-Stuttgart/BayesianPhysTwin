from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_query_calibration import _IDS, _fit

from bayesian_phystwin.query_calibration import (
    QueryCalibrationV1,
    calibrate_query_covariance,
    group_mahalanobis_nonconformity,
    load_query_calibration,
    query_group_is_covered,
    save_query_calibration,
)


def _calibration_with_quantile(quantile: float) -> QueryCalibrationV1:
    scores = np.arange(1.0, 10.0, dtype=np.float64)
    scores[-1] = quantile
    return QueryCalibrationV1(
        **_IDS,
        calibration_group_ids=tuple(f"object-{index:02d}" for index in range(9)),
        calibration_group_scores=scores,
        nominal_coverage=0.9,
        finite_sample_rank=9,
        conformal_quantile=quantile,
        covariance_scale=1.0,
        isotropic_variance=0.0,
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
    )


def test_retained_scores_and_deployed_covariances_are_irreversibly_immutable() -> None:
    calibration = _fit()
    deployed = calibrate_query_covariance(np.eye(2), calibration)

    for array in (calibration.calibration_group_scores, deployed):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_constructor_detaches_scores_before_building_content_identity() -> None:
    scores = np.arange(1.0, 10.0, dtype=np.float64)
    calibration = QueryCalibrationV1(
        **_IDS,
        calibration_group_ids=tuple(f"object-{index:02d}" for index in range(9)),
        calibration_group_scores=scores,
        nominal_coverage=0.9,
        finite_sample_rank=9,
        conformal_quantile=9.0,
        covariance_scale=1.0,
        isotropic_variance=0.0,
        predictor_frozen_before_scores=True,
        calibration_outcomes_used_for_selection=False,
    )
    artifact_id = calibration.artifact_id

    scores[:] = 0.0

    np.testing.assert_array_equal(
        calibration.calibration_group_scores,
        np.arange(1.0, 10.0),
    )
    assert calibration.artifact_id == artifact_id


def test_atomic_save_is_idempotent_and_refuses_a_different_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query-calibration.json"
    calibration = _fit()
    save_query_calibration(calibration, path)
    retained_bytes = path.read_bytes()

    save_query_calibration(calibration, path)

    assert path.read_bytes() == retained_bytes
    assert not list(tmp_path.glob(".query-calibration.json.*.tmp"))
    with pytest.raises(ValueError, match="refusing to replace"):
        save_query_calibration(_fit(covariance_scale=4.0), path)
    assert path.read_bytes() == retained_bytes


def test_save_does_not_replace_corrupt_or_nonregular_destinations(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        save_query_calibration(_fit(), corrupt)
    assert corrupt.read_text(encoding="utf-8") == "not-json"

    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        save_query_calibration(_fit(), directory)


def test_public_consumers_require_the_validated_calibration_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="QueryCalibrationV1"):
        calibrate_query_covariance(
            np.eye(1),
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="QueryCalibrationV1"):
        query_group_is_covered(
            np.zeros(1),
            np.eye(1),
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="QueryCalibrationV1"):
        save_query_calibration(
            object(),  # type: ignore[arg-type]
            tmp_path / "wrong.json",
        )


def test_load_normalizes_missing_and_invalid_json_failures(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unreadable"):
        load_query_calibration(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_query_calibration(invalid)


def test_nonfinite_transforms_and_scores_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite after the frozen transform"):
        group_mahalanobis_nonconformity(
            np.ones(1),
            np.asarray([[1e308]]),
            covariance_scale=1e308,
        )
    with pytest.raises(ValueError, match="Mahalanobis score must be finite"):
        group_mahalanobis_nonconformity(
            np.asarray([1e308]),
            np.asarray([[1e-308]]),
        )


def test_conformal_covariance_overflow_fails_closed() -> None:
    with pytest.raises(ValueError, match="multiplier must be finite"):
        calibrate_query_covariance(
            np.eye(1),
            _calibration_with_quantile(1e308),
        )
    with pytest.raises(ValueError, match="calibrated covariance 0 must be finite"):
        calibrate_query_covariance(
            2.0 * np.eye(1),
            _calibration_with_quantile(1e154),
        )


def test_zero_calibration_quantile_produces_a_deterministic_degenerate_region() -> None:
    calibration = _calibration_with_quantile(0.0)
    calibrated = calibrate_query_covariance(np.eye(2), calibration)

    np.testing.assert_array_equal(calibrated, np.zeros((2, 2)))
    assert query_group_is_covered(np.zeros(2), np.eye(2), calibration)
    assert not query_group_is_covered(np.asarray([1e-12, 0.0]), np.eye(2), calibration)
    with pytest.raises(ValueError):
        calibrated.setflags(write=True)
