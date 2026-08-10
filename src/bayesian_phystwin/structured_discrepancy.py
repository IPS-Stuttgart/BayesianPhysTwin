"""Structured low-rank endpoint beliefs for deformable-object discrepancy.

This additive development interface represents an observable discrepancy field
with shared spatial modes and a marginal-preserving local variance remainder. It
does not identify a simulator-state correction, and raw covariance is not a
calibration claim.
"""

from ._structured_discrepancy_contracts import (
    STRUCTURED_DISCREPANCY_CLAIM_BOUNDARY,
    STRUCTURED_DISCREPANCY_CONTRACT_VERSION,
    STRUCTURED_DISCREPANCY_COVARIANCE_SEMANTICS,
    STRUCTURED_DISCREPANCY_EVIDENCE_SEMANTICS,
    StructuredDiscrepancyConfigV1,
    StructuredDiscrepancyPosteriorV1,
    StructuredDiscrepancyPredictionV1,
    StructuredDiscrepancyQueryMomentsV1,
)
from ._structured_discrepancy_inference import (
    infer_structured_discrepancy,
    predict_structured_discrepancy,
)
from ._structured_discrepancy_query import (
    structured_discrepancy_query_moments,
)

__all__ = [
    "STRUCTURED_DISCREPANCY_CLAIM_BOUNDARY",
    "STRUCTURED_DISCREPANCY_CONTRACT_VERSION",
    "STRUCTURED_DISCREPANCY_COVARIANCE_SEMANTICS",
    "STRUCTURED_DISCREPANCY_EVIDENCE_SEMANTICS",
    "StructuredDiscrepancyConfigV1",
    "StructuredDiscrepancyPosteriorV1",
    "StructuredDiscrepancyPredictionV1",
    "StructuredDiscrepancyQueryMomentsV1",
    "infer_structured_discrepancy",
    "predict_structured_discrepancy",
    "structured_discrepancy_query_moments",
]
