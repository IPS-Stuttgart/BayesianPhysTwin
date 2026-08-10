"""Graph-modal Bayesian dynamics for predictive readout discrepancy.

This module models discrepancy separately from latent physical state. A graph
basis couples nodes spatially, while modal position and velocity coefficients
carry recursive uncertainty through time. Observation updates use one robust
Student-t weight per declared correlation group and fail back to the exact
predicted belief whenever numerical admission or plausibility checks fail.
"""

from ._graph_dynamic_discrepancy_common import (
    GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY,
    GRAPH_DYNAMIC_DISCREPANCY_SCHEMA,
    GRAPH_DYNAMIC_DISCREPANCY_VERSION,
    GraphDynamicDiscrepancyConfigV1,
)
from ._graph_dynamic_discrepancy_contract import (
    GraphDynamicDiscrepancyBeliefV1,
    GraphDynamicDiscrepancyForecastV1,
)
from ._graph_dynamic_discrepancy_filter import fit_graph_dynamic_discrepancy

__all__ = [
    "GRAPH_DYNAMIC_DISCREPANCY_BOUNDARY",
    "GRAPH_DYNAMIC_DISCREPANCY_SCHEMA",
    "GRAPH_DYNAMIC_DISCREPANCY_VERSION",
    "GraphDynamicDiscrepancyBeliefV1",
    "GraphDynamicDiscrepancyConfigV1",
    "GraphDynamicDiscrepancyForecastV1",
    "fit_graph_dynamic_discrepancy",
]
