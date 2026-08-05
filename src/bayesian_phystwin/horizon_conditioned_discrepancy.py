"""Public source-calibrated horizon dynamics for discrepancy endpoints.

The historical constant-mean endpoint predictor remains unchanged. This module
adds content-addressed source-group fitting, horizon-dependent mean retention,
stationary discrepancy uncertainty, and strictly positive process growth.
"""

from ._horizon_discrepancy_common import (
    HORIZON_DISCREPANCY_CALIBRATION_SCHEMA,
    HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS,
    HORIZON_DISCREPANCY_CALIBRATION_VERSION,
)
from ._horizon_discrepancy_contract import (
    HorizonConditionedEndpointPredictionV1,
    HorizonDiscrepancyCalibrationV1,
    load_horizon_discrepancy_calibration,
    save_horizon_discrepancy_calibration,
)
from ._horizon_discrepancy_fit import fit_horizon_discrepancy_calibration
from ._horizon_discrepancy_prediction import (
    mean_retention_at_horizon,
    predict_horizon_conditioned_endpoint,
)

__all__ = [
    "HORIZON_DISCREPANCY_CALIBRATION_SCHEMA",
    "HORIZON_DISCREPANCY_CALIBRATION_SEMANTICS",
    "HORIZON_DISCREPANCY_CALIBRATION_VERSION",
    "HorizonConditionedEndpointPredictionV1",
    "HorizonDiscrepancyCalibrationV1",
    "fit_horizon_discrepancy_calibration",
    "load_horizon_discrepancy_calibration",
    "mean_retention_at_horizon",
    "predict_horizon_conditioned_endpoint",
    "save_horizon_discrepancy_calibration",
]
