"""Robust model averaging across persistent, local-level, and trend endpoints.

The historical endpoint model average remains unchanged. This additive v2
surface includes the strongest simple deterministic comparator as an exact
sample-and-hold component and adds damped local-trend components whose forecast
mean and covariance both depend on the requested horizon.
"""

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
from ._dynamic_endpoint_filter import infer_dynamic_endpoint_model_average
from ._dynamic_endpoint_prediction import predict_dynamic_endpoint_model_average

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
