"""Broken-mechanism physicality certificate for chronological transport.

The contract is an additive, target-closed analysis layer over
:mod:`cross_action_transport_v2`. It asks whether an already supported guarded
physical prediction separates from four source constructions that deliberately
break the registered physical relation while preserving the target query.
Complete physical sessions remain the independent statistical units.
"""

from bayesian_phystwin_experiments._cross_action_physicality_common_v1 import (
    CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY,
    CROSS_ACTION_PHYSICALITY_SCHEMA,
    CROSS_ACTION_PHYSICALITY_SEMANTICS,
    CROSS_ACTION_PHYSICALITY_VERSION,
    FAMILYWISE_METHOD,
    FAMILYWISE_METHOD_ID,
    REQUIRED_PLACEBO_POLICIES,
    BrokenMechanismPolicy,
    PhysicalityDecision,
)
from bayesian_phystwin_experiments._cross_action_physicality_protocol_v1 import (
    CrossActionPhysicalityProtocolV1,
)
from bayesian_phystwin_experiments._cross_action_physicality_records_v1 import (
    PlaceboConstructionV1,
    PlaceboScoreRowV1,
    SealedPlaceboPredictionV1,
)
from bayesian_phystwin_experiments._cross_action_physicality_result_v1 import (
    CrossActionPhysicalityResultV1,
    PlaceboContrastSummaryV1,
)

__all__ = [
    "CROSS_ACTION_PHYSICALITY_CLAIM_BOUNDARY",
    "CROSS_ACTION_PHYSICALITY_SCHEMA",
    "CROSS_ACTION_PHYSICALITY_SEMANTICS",
    "CROSS_ACTION_PHYSICALITY_VERSION",
    "FAMILYWISE_METHOD",
    "FAMILYWISE_METHOD_ID",
    "REQUIRED_PLACEBO_POLICIES",
    "BrokenMechanismPolicy",
    "CrossActionPhysicalityProtocolV1",
    "CrossActionPhysicalityResultV1",
    "PhysicalityDecision",
    "PlaceboConstructionV1",
    "PlaceboContrastSummaryV1",
    "PlaceboScoreRowV1",
    "SealedPlaceboPredictionV1",
]
