"""Outcome-blind recovery amendment after the Slingshot v2 import failure."""

from __future__ import annotations

import copy
from typing import Any, cast

from ._portable_contracts import content_id
from .query_portfolio_replication_v1 import QueryOutcomeV1
from .query_portfolio_replication_v2 import protocol as protocol_v2
from .query_portfolio_replication_v2 import score as score_v2

SCHEMA = "bayesian_phystwin.query_portfolio_replication"
VERSION = 3
SLINGSHOT_WORLD_SEEDS = {"calibration": 263_301, "evaluation": 263_302}
SLINGSHOT_SENSOR_SEEDS = {"calibration": 263_303, "evaluation": 263_304}
PARENT_PROTOCOL_ID = "511f9451c582ded23cab24bbf2721b00c5079259a044aaf0b8b8fe45f4ce9dcd"
TERMINAL_SLINGSHOT_V2_FAILURE_ID = (
    "b0409889fb5b14d0f7eb558f8aa77965f6fbcb6d703909f465fe9ca26f496cad"
)


def protocol() -> dict[str, Any]:
    """Return the recovery design frozen before any v3 world generation."""

    value = copy.deepcopy(cast(dict[str, Any], protocol_v2()))
    if value.pop("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError("portfolio v2 protocol changed")
    value["version"] = VERSION
    value["role"] = "prospective_two_query_public_simulator_replication_recovery_v3"
    value["parent_protocol_id"] = PARENT_PROTOCOL_ID
    value["world_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_WORLD_SEEDS
    value["sensor_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_SENSOR_SEEDS
    value["recovery"] = {
        "terminal_slingshot_v2_failure_id": TERMINAL_SLINGSHOT_V2_FAILURE_ID,
        "terminal_stage": "calibration-prefixes",
        "ordinary_prefix_seals": 0,
        "ordinary_future_seals": 0,
        "outcomes_opened": False,
        "worlds_reused": False,
        "scientific_change": False,
        "repair": "add_exact_staged_dlolab_experiments_to_worker_pythonpath",
        "worker_import_preflight_required": True,
    }
    value["outcomes_opened"] = False
    value["protocol_id"] = content_id(value)
    return value


def score(outcomes: dict[str, QueryOutcomeV1]) -> dict[str, Any]:
    """Apply the unchanged joint scoring rule to the v3 rosters."""

    value = cast(dict[str, Any], score_v2(outcomes))
    value.pop("artifact_id")
    value["version"] = VERSION
    value["protocol_id"] = protocol()["protocol_id"]
    value["artifact_id"] = content_id(value)
    return value
