"""Prediction operations for source-calibrated horizon discrepancy dynamics."""

from __future__ import annotations

import numpy as np

from ._canonical_contracts import genuine_integer
from ._horizon_discrepancy_contract import (
    HorizonConditionedEndpointPredictionV1,
    HorizonDiscrepancyCalibrationV1,
)
from .endpoint_model_average import ModelAveragedEndpointPosteriorV1


def mean_retention_at_horizon(
    calibration: HorizonDiscrepancyCalibrationV1, horizon_steps: int
) -> float:
    """Return the frozen discrepancy-mean retention at one horizon."""

    if not isinstance(calibration, HorizonDiscrepancyCalibrationV1):
        raise TypeError("calibration must be a HorizonDiscrepancyCalibrationV1")
    horizon = genuine_integer(horizon_steps, name="horizon_steps", minimum=0)
    half_life = calibration.mean_reversion_half_life_steps
    if horizon == 0 or half_life is None:
        return 1.0
    floor = calibration.minimum_mean_retention
    return float(floor + (1.0 - floor) * 2.0 ** (-horizon / half_life))


def predict_horizon_conditioned_endpoint(
    posterior: ModelAveragedEndpointPosteriorV1,
    calibration: HorizonDiscrepancyCalibrationV1,
    *,
    horizon_steps: int,
) -> HorizonConditionedEndpointPredictionV1:
    """Propagate endpoint moments with source-frozen horizon dynamics."""

    if not isinstance(posterior, ModelAveragedEndpointPosteriorV1):
        raise TypeError("posterior must be a ModelAveragedEndpointPosteriorV1")
    horizon = genuine_integer(horizon_steps, name="horizon_steps", minimum=0)
    retention = mean_retention_at_horizon(calibration, horizon)
    if horizon == 0:
        return HorizonConditionedEndpointPredictionV1(
            mean_m=posterior.mean_m,
            covariance_m2=posterior.covariance_m2,
            component_weights=posterior.component_weights,
            component_mean_m=posterior.component_mean_m,
            component_variance_m2=posterior.component_variance_m2,
            additional_axis_variance_m2=np.zeros(3),
            horizon_steps=0,
            mean_retention=1.0,
            calibration_id=calibration.artifact_id,
        )

    component_mean = retention * posterior.component_mean_m
    component_variance = retention**2 * posterior.component_variance_m2
    component_variance += (
        horizon
        * calibration.component_process_variance_scale
        * posterior.component_process_variance_m2[:, None]
    )
    mean = np.einsum("nk,knc->nc", posterior.component_weights, component_mean)
    centered = component_mean - mean[None, :, :]
    outer = centered[:, :, :, None] * centered[:, :, None, :]
    within = component_variance[:, :, None, None] * np.eye(3)
    covariance = np.einsum(
        "nk,knij->nij", posterior.component_weights, within + outer
    )
    additional = (
        (1.0 - retention**2) * np.square(calibration.stationary_std_m)
        + horizon
        * np.square(calibration.additional_process_std_m_per_sqrt_step)
    )
    covariance += np.diag(additional)[None, :, :]
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    return HorizonConditionedEndpointPredictionV1(
        mean_m=mean,
        covariance_m2=covariance,
        component_weights=posterior.component_weights,
        component_mean_m=component_mean,
        component_variance_m2=component_variance,
        additional_axis_variance_m2=additional,
        horizon_steps=horizon,
        mean_retention=retention,
        calibration_id=calibration.artifact_id,
    )
