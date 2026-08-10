"""Source-only tournament adapter for graph-modal discrepancy forecasts.

Forecast generation and outcome scoring are intentionally separate. The seal
constructor cannot receive a scored target; the score constructor accepts only
an already content-addressed complete prediction roster.
"""

from ._graph_dynamic_tournament_common import (
    GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY,
    GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_VERSION,
    GRAPH_DYNAMIC_TOURNAMENT_FAMILY,
    GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_VERSION,
    GRAPH_DYNAMIC_TOURNAMENT_SCORE_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_SCORE_VERSION,
    GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_SCHEMA,
    GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_VERSION,
)
from ._graph_dynamic_tournament_contract import (
    GraphDynamicTournamentPredictionBundleV1,
    GraphDynamicTournamentPredictionV1,
    GraphDynamicTournamentScoringPolicyV1,
    build_graph_dynamic_tournament_prediction_bundle,
    seal_graph_dynamic_tournament_prediction,
)
from ._graph_dynamic_tournament_score import (
    GraphDynamicTournamentScoredBundleV1,
    candidate_spec_dict,
    score_graph_dynamic_tournament_prediction_bundle,
    tournament_record_dict,
)

__all__ = [
    "GRAPH_DYNAMIC_TOURNAMENT_BOUNDARY",
    "GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_BUNDLE_VERSION",
    "GRAPH_DYNAMIC_TOURNAMENT_FAMILY",
    "GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_PREDICTION_VERSION",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORE_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORE_VERSION",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_SCHEMA",
    "GRAPH_DYNAMIC_TOURNAMENT_SCORING_POLICY_VERSION",
    "GraphDynamicTournamentPredictionBundleV1",
    "GraphDynamicTournamentPredictionV1",
    "GraphDynamicTournamentScoredBundleV1",
    "GraphDynamicTournamentScoringPolicyV1",
    "build_graph_dynamic_tournament_prediction_bundle",
    "candidate_spec_dict",
    "score_graph_dynamic_tournament_prediction_bundle",
    "seal_graph_dynamic_tournament_prediction",
    "tournament_record_dict",
]
