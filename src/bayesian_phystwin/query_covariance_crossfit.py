"""Group-cross-fitted structured covariance transforms for physical queries.

The public facade separates immutable contracts, selection, and scoring while
leaving the package root unchanged.
"""

from ._query_covariance_crossfit_common import (
    QUERY_COVARIANCE_CROSSFIT_CLAIM_BOUNDARY,
    QUERY_COVARIANCE_CROSSFIT_SCHEMA,
    QUERY_COVARIANCE_CROSSFIT_SCORE,
    QUERY_COVARIANCE_CROSSFIT_VERSION,
    StructuredQueryCovarianceCandidateV1,
    StructuredQueryCovarianceTransformV1,
    apply_structured_query_covariance,
)
from ._query_covariance_crossfit_scoring import (
    QueryCovarianceGroupDiagnosticsV1,
    group_gaussian_energy_score,
    score_query_covariance_group,
)
from ._query_covariance_crossfit_selection import (
    QueryCovarianceCrossFitV1,
    fit_cross_fitted_query_covariance,
)

__all__ = [
    "QUERY_COVARIANCE_CROSSFIT_CLAIM_BOUNDARY",
    "QUERY_COVARIANCE_CROSSFIT_SCHEMA",
    "QUERY_COVARIANCE_CROSSFIT_SCORE",
    "QUERY_COVARIANCE_CROSSFIT_VERSION",
    "QueryCovarianceCrossFitV1",
    "QueryCovarianceGroupDiagnosticsV1",
    "StructuredQueryCovarianceCandidateV1",
    "StructuredQueryCovarianceTransformV1",
    "apply_structured_query_covariance",
    "fit_cross_fitted_query_covariance",
    "group_gaussian_energy_score",
    "score_query_covariance_group",
]
