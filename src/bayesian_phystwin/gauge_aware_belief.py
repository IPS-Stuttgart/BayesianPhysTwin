"""Gauge- and bias-aware Bayesian updates from unfused 4-D factors.

The estimator keeps Prob4D window gauges as nuisance variables, caps shared
factor information, and updates only query-relevant physical state directions
that remain identifiable beyond gauge and observation-bias uncertainty.
"""

from ._gauge_aware_contracts import (
    GaugeAwareBeliefConfig,
    GaugeAwareBeliefResult,
    GaugeAwareObservationBatch,
    GaugeAwareSelection,
)
from ._gauge_aware_solver import (
    decode_gauge_aware_query,
    select_gauge_aware_candidate,
    update_gauge_aware_belief,
)

__all__ = [
    "GaugeAwareBeliefConfig",
    "GaugeAwareBeliefResult",
    "GaugeAwareObservationBatch",
    "GaugeAwareSelection",
    "decode_gauge_aware_query",
    "select_gauge_aware_candidate",
    "update_gauge_aware_belief",
]
