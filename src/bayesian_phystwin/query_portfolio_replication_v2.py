"""Outcome-blind recovery amendment for portfolio replication v1."""

from __future__ import annotations

import copy
from typing import Any, cast

from ._portable_contracts import content_id
from .query_portfolio_replication_v1 import QueryOutcomeV1
from .query_portfolio_replication_v1 import protocol as protocol_v1
from .query_portfolio_replication_v1 import score as score_v1

SCHEMA = "bayesian_phystwin.query_portfolio_replication"
VERSION = 2
SLINGSHOT_WORLD_SEEDS = {"calibration": 263_201, "evaluation": 263_202}
SLINGSHOT_SENSOR_SEEDS = {"calibration": 263_203, "evaluation": 263_204}
PARENT_PROTOCOL_ID = "7a741884b779d080f1e041a2824686cced29e33616481c74e1ca47d71957969e"
TERMINAL_SLINGSHOT_V1_FAILURE_ID = (
    "78f9c3d96d6fc19cea8526bcd4324bbf0a4cfb4d01f4ea491ba16de400610df6"
)


def protocol() -> dict[str, Any]:
    """Return the recovery design frozen before any v2 world generation."""

    value = copy.deepcopy(cast(dict[str, Any], protocol_v1()))
    if value.pop("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError("portfolio v1 protocol changed")
    value["version"] = VERSION
    value["role"] = "prospective_two_query_public_simulator_replication_recovery"
    value["parent_protocol_id"] = PARENT_PROTOCOL_ID
    value["world_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_WORLD_SEEDS
    value["sensor_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_SENSOR_SEEDS
    value["recovery"] = {
        "terminal_slingshot_v1_failure_id": TERMINAL_SLINGSHOT_V1_FAILURE_ID,
        "terminal_stage": "calibration-prefixes",
        "ordinary_prefix_seals": 0,
        "ordinary_future_seals": 0,
        "outcomes_opened": False,
        "worlds_reused": False,
        "scientific_change": False,
        "repair": "translate_canonical_library_path_to_staged_worker_path",
    }
    value["outcomes_opened"] = False
    value["protocol_id"] = content_id(value)
    return value


def score(outcomes: dict[str, QueryOutcomeV1]) -> dict[str, Any]:
    """Apply the unchanged v1 joint scoring rule to the v2 rosters."""

    value = cast(dict[str, Any], score_v1(outcomes))
    value.pop("artifact_id")
    value["version"] = VERSION
    value["protocol_id"] = protocol()["protocol_id"]
    value["artifact_id"] = content_id(value)
    return value
