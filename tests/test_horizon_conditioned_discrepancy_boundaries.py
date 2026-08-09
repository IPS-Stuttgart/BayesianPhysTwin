from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from bayesian_phystwin._horizon_discrepancy_common import (
    axis_vector,
    finite_real,
    horizon_vector,
    probability,
)
from bayesian_phystwin.horizon_conditioned_discrepancy import (
    HORIZON_DISCREPANCY_CALIBRATION_SCHEMA,
    HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS,
    HorizonConditionedEndpointPredictionV1,
    HorizonDiscrepancyCalibrationV1,
    save_horizon_discrepancy_calibration,
)


def _calibration(**updates: object) -> HorizonDiscrepancyCalibrationV1:
    values: dict[str, object] = {
        "source_group_ids": ("object-a", "object-b"),
        "source_summary_sha256": "a" * 64,
        "horizon_steps": (1, 4),
        "mean_reversion_half_life_steps": 8.0,
        "minimum_mean_retention": 0.25,
        "stationary_std_m": np.array([0.003, 0.004, 0.005]),
        "additional_process_std_m_per_sqrt_step": np.array([0.0001, 0.0002, 0.0003]),
    }
    values.update(updates)
    return HorizonDiscrepancyCalibrationV1(**values)  # type: ignore[arg-type]


def _prediction(**updates: object) -> HorizonConditionedEndpointPredictionV1:
    values: dict[str, object] = {
        "mean_m": np.zeros((2, 3)),
        "covariance_m2": np.repeat(np.eye(3)[None, :, :], 2, axis=0),
        "component_weights": np.array([[0.5, 0.5], [0.4, 0.6]]),
        "component_mean_m": np.zeros((2, 2, 3)),
        "component_variance_m2": np.full((2, 2), 0.1),
        "additional_axis_variance_m2": np.array([0.01, 0.02, 0.03]),
        "horizon_steps": 4,
        "mean_retention": 0.75,
        "calibration_id": "b" * 64,
    }
    values.update(updates)
    return HorizonConditionedEndpointPredictionV1(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("operation", "message"),
    (
        (lambda: finite_real(True, name="value"), "finite real"),
        (lambda: finite_real(np.nan, name="value"), "finite real"),
        (lambda: probability(1.1, name="probability"), r"\[0, 1\]"),
        (lambda: axis_vector(["x", "y", "z"], name="axis"), "length-3"),
        (lambda: axis_vector([1.0, 2.0], name="axis"), "length-3"),
        (lambda: horizon_vector([1.0, 2.0], allow_zero=False), "integers"),
    ),
)
def test_common_validation_helpers_fail_closed(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_calibration_mapping_and_source_boundary_fail_closed() -> None:
    with pytest.raises(ValueError, match="source outcomes"):
        _calibration(source_outcomes_used=False)

    with pytest.raises(ValueError, match="JSON object"):
        HorizonDiscrepancyCalibrationV1.from_mapping([])

    record = _calibration().to_record()
    record["schema"] = "other-schema"
    with pytest.raises(ValueError, match="unsupported.*schema"):
        HorizonDiscrepancyCalibrationV1.from_mapping(record)

    record = _calibration().to_record()
    record["semantics"] = "other-semantics"
    with pytest.raises(ValueError, match="semantics changed"):
        HorizonDiscrepancyCalibrationV1.from_mapping(record)

    assert (
        HORIZON_DISCREPANCY_CALIBRATION_SCHEMA in _calibration().descriptor()["schema"]
    )
    assert (
        _calibration().descriptor()["semantics"]
        == HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS
    )


def test_prediction_contract_rejects_invalid_moment_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="mean_m"):
        _prediction(mean_m=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="covariance_m2 must have shape"):
        _prediction(covariance_m2=np.zeros((2, 2, 2)))

    nonfinite_mean = np.zeros((2, 3))
    nonfinite_mean[0, 0] = np.nan
    with pytest.raises(ValueError, match="moments must be finite"):
        _prediction(mean_m=nonfinite_mean)

    nonsymmetric = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    nonsymmetric[0, 0, 1] = 0.5
    with pytest.raises(ValueError, match="symmetric"):
        _prediction(covariance_m2=nonsymmetric)

    indefinite = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    indefinite[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        _prediction(covariance_m2=indefinite)


def test_prediction_contract_rejects_invalid_component_structure() -> None:
    with pytest.raises(ValueError, match="component_weights shape"):
        _prediction(component_weights=np.ones((2, 2, 1)))
    with pytest.raises(ValueError, match="row-normalized"):
        _prediction(component_weights=np.array([[0.8, 0.8], [0.4, 0.6]]))
    with pytest.raises(ValueError, match="component_mean_m shape"):
        _prediction(component_mean_m=np.zeros((2, 1, 3)))
    with pytest.raises(ValueError, match="component_variance_m2 shape"):
        _prediction(component_variance_m2=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="nonnegative length 3"):
        _prediction(additional_axis_variance_m2=np.array([0.1, -0.1, 0.2]))

    nonfinite_component_mean = np.zeros((2, 2, 3))
    nonfinite_component_mean[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="component values must be finite"):
        _prediction(component_mean_m=nonfinite_component_mean)

    negative_component_variance = np.full((2, 2), 0.1)
    negative_component_variance[0, 0] = -0.1
    with pytest.raises(ValueError, match="must be nonnegative"):
        _prediction(component_variance_m2=negative_component_variance)


def test_save_rejects_noncalibration_object(tmp_path) -> None:
    with pytest.raises(TypeError, match="HorizonDiscrepancyCalibrationV1"):
        save_horizon_discrepancy_calibration(  # type: ignore[arg-type]
            object(), tmp_path / "invalid.json"
        )
