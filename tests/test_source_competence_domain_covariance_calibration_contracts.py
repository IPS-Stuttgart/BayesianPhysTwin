from __future__ import annotations

import numpy as np
import pytest
from domain_covariance_calibration_test_helpers import (
    _certificate,
    _inputs,
)

from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationConfigV1,
    fit_domain_covariance_calibration,
)


def test_singleton_and_disabled_grids_are_explicit() -> None:
    config = DomainCovarianceCalibrationConfigV1(
        minimum_scale=2.0,
        maximum_scale=2.0,
        scale_grid_size=1,
        floor_grid_size=0,
    )
    assert config.scale_grid() == (2.0,)
    assert config.floor_ratio_grid() == (0.0,)
    one_floor = DomainCovarianceCalibrationConfigV1(
        floor_grid_size=1,
        minimum_positive_floor_ratio=0.25,
    )
    assert one_floor.floor_ratio_grid() == (0.0, 0.25)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_scale", 0.5),
        ("maximum_floor_ratio", -1.0),
        ("symmetry_tolerance", True),
        ("score_tolerance", np.nan),
        ("scale_grid_size", 1.5),
        ("log_loss_ratio_clip", 0.0),
    ),
)
def test_config_rejects_noncanonical_numeric_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        DomainCovarianceCalibrationConfigV1(  # type: ignore[arg-type]
            **{field: value}
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("statistical_unit", "", "canonical string"),
        ("group_ids", "not-a-sequence", "sequence of strings"),
        ("group_ids", 7, "sequence of strings"),
        ("group_ids", (), "must not be empty"),
        ("residuals", (np.asarray([1.0, 2.0]),) * 6, "2-dimensional"),
    ),
)
def test_fit_rejects_noncanonical_top_level_inputs(
    field: str,
    value: object,
    match: str,
) -> None:
    arguments = _inputs()
    arguments[field] = value
    with pytest.raises(ValueError, match=match):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]


def test_fit_rejects_nonfinite_residual_and_wrong_config_types() -> None:
    arguments = _inputs()
    residuals = list(arguments["residuals"])
    malformed = np.asarray(residuals[0]).copy()
    malformed[0, 0] = np.nan
    residuals[0] = malformed
    arguments["residuals"] = tuple(residuals)
    with pytest.raises(ValueError, match="finite"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="DomainCovarianceCalibrationConfigV1"):
        _certificate(config=object())
    with pytest.raises(TypeError, match="CalibrationDomainGuardConfigV1"):
        _certificate(guard_config=object())


def test_group_shapes_and_symmetry_fail_closed() -> None:
    arguments = _inputs()
    samples = list(arguments["sample_ids"])
    samples[0] = samples[0][:-1]
    arguments["sample_ids"] = tuple(samples)
    with pytest.raises(ValueError, match="equal lengths"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

    arguments = _inputs()
    residuals = list(arguments["residuals"])
    residuals[1] = np.asarray(residuals[1])[:, :2]
    arguments["residuals"] = tuple(residuals)
    with pytest.raises(ValueError, match="same dimension"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

    arguments = _inputs()
    covariances = list(arguments["covariances"])
    covariances[0] = np.asarray(covariances[0])[:, :2, :2]
    arguments["covariances"] = tuple(covariances)
    with pytest.raises(ValueError, match="must have shape"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

    arguments = _inputs()
    covariances = list(arguments["covariances"])
    asymmetric = np.asarray(covariances[0]).copy()
    asymmetric[0, 0, 1] = 0.2
    covariances[0] = asymmetric
    arguments["covariances"] = tuple(covariances)
    with pytest.raises(ValueError, match="symmetric"):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]
