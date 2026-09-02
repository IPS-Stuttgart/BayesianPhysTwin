"""Outcome-blind recovery after the Slingshot v4 transitive-count defect."""

from __future__ import annotations

import copy
from typing import Any, cast

from ._portable_contracts import content_id
from .query_portfolio_replication_v1 import QueryOutcomeV1
from .query_portfolio_replication_v4 import protocol as protocol_v4
from .query_portfolio_replication_v4 import score as score_v4

VERSION = 5
SLINGSHOT_WORLD_SEEDS = {"calibration": 263_501, "evaluation": 263_502}
SLINGSHOT_SENSOR_SEEDS = {"calibration": 263_503, "evaluation": 263_504}
PARENT_PROTOCOL_ID = "595ea3752b83ad8403555bd5f237a8b0ccf764b7e1bfa094fe8a36664182a7c2"
TERMINAL_SLINGSHOT_V4_FAILURE_ID = (
    "b08af24f653f84ae06f5f61da3a2e203fcb52f885b0b8326213a0c2e101bff82"
)


def protocol() -> dict[str, Any]:
    value = copy.deepcopy(cast(dict[str, Any], protocol_v4()))
    if value.pop("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError("portfolio v4 protocol changed")
    value["version"] = VERSION
    value["role"] = "prospective_two_query_public_simulator_replication_recovery_v5"
    value["parent_protocol_id"] = PARENT_PROTOCOL_ID
    value["world_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_WORLD_SEEDS
    value["sensor_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_SENSOR_SEEDS
    value["recovery"] = {
        "terminal_slingshot_v4_failure_id": TERMINAL_SLINGSHOT_V4_FAILURE_ID,
        "terminal_stage": "evaluation-decision-barrier",
        "calibration_prefixes_sealed": 128,
        "calibration_futures_sealed": 128 * 7,
        "evaluation_prefixes_sealed": 320,
        "evaluation_candidates_sealed": 320,
        "evaluation_futures_sealed": 0,
        "evaluation_outcomes_opened": False,
        "partial_score_authorized": False,
        "worlds_reused": False,
        "scientific_change": False,
        "repair": "bind_transitive_v2_helpers_to_registered_320_world_count",
        "shape_preflight_registered_worlds": 0,
        "shape_preflight_actions": 0,
    }
    value["protocol_id"] = content_id(value)
    return value


def score(outcomes: dict[str, QueryOutcomeV1]) -> dict[str, Any]:
    value = cast(dict[str, Any], score_v4(outcomes))
    value.pop("artifact_id")
    value["version"] = VERSION
    value["protocol_id"] = protocol()["protocol_id"]
    value["artifact_id"] = content_id(value)
    return value
