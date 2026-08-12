"""Robust model averaging across persistent, local-level, and trend endpoints.

The historical endpoint model average remains unchanged. This additive v2
surface includes the strongest simple deterministic comparator as an exact
sample-and-hold component and adds damped local-trend components whose forecast
mean and covariance both depend on the requested horizon.
"""

from __future__ import annotations

import numpy as np

from ._dynamic_endpoint_components import (
    DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2,
    DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION,
    DampedTrendEndpointComponentV2,
    DynamicEndpointComponentV2,
    DynamicEndpointModelAverageConfigV2,
    DynamicEndpointNumericalError,
    EvidencePoolingV2,
    PersistenceEndpointComponentV2,
    component_kind,
)
from ._dynamic_endpoint_contract import (
    DynamicEndpointPosteriorV2,
    DynamicEndpointPredictionV2,
)
from ._dynamic_endpoint_filter import (
    infer_dynamic_endpoint_model_average as _infer_dynamic_endpoint_model_average,
)
from ._dynamic_endpoint_prediction import predict_dynamic_endpoint_model_average


def infer_dynamic_endpoint_model_average(
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: DynamicEndpointModelAverageConfigV2 | None = None,
    observation_variance_m2: np.ndarray | None = None,
) -> DynamicEndpointPosteriorV2:
    """Infer a dynamic endpoint with optional metric observation variance."""

    try:
        raw_residual = np.asarray(residual_m)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("residual_m must contain real numeric values") from error
    if raw_residual.dtype.kind not in "iuf":
        raise ValueError("residual_m must contain real numeric values")
    return _infer_dynamic_endpoint_model_average(
        np.asarray(raw_residual, dtype=np.float64),
        valid,
        end_frame=end_frame,
        config=config,
        observation_variance_m2=observation_variance_m2,
    )


__all__ = [
    "DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2",
    "DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION",
    "DampedTrendEndpointComponentV2",
    "DynamicEndpointComponentV2",
    "DynamicEndpointModelAverageConfigV2",
    "DynamicEndpointNumericalError",
    "DynamicEndpointPosteriorV2",
    "DynamicEndpointPredictionV2",
    "EvidencePoolingV2",
    "PersistenceEndpointComponentV2",
    "component_kind",
    "infer_dynamic_endpoint_model_average",
    "predict_dynamic_endpoint_model_average",
]
