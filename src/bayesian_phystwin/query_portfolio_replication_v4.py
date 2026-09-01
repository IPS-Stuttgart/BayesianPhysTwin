"""Outcome-blind recovery amendment after the sparse Slingshot v3 asset tree."""

from __future__ import annotations

import copy
from typing import Any, cast

from ._portable_contracts import content_id
from .query_portfolio_replication_v1 import QueryOutcomeV1
from .query_portfolio_replication_v3 import protocol as protocol_v3
from .query_portfolio_replication_v3 import score as score_v3

VERSION = 4
SLINGSHOT_WORLD_SEEDS = {"calibration": 263_401, "evaluation": 263_402}
SLINGSHOT_SENSOR_SEEDS = {"calibration": 263_403, "evaluation": 263_404}
PARENT_PROTOCOL_ID = "c1e507fa445bca097c1b3d1f502fa9042584454dc12233d481c90f8223fc135e"
TERMINAL_SLINGSHOT_V3_FAILURE_ID = (
    "d4af5d766a1dfb967ba4abc64f100444c7166fe1e2d75fe0eedd5dca1e17457b"
)


def protocol() -> dict[str, Any]:
    value = copy.deepcopy(cast(dict[str, Any], protocol_v3()))
    if value.pop("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError("portfolio v3 protocol changed")
    value["version"] = VERSION
    value["role"] = "prospective_two_query_public_simulator_replication_recovery_v4"
    value["parent_protocol_id"] = PARENT_PROTOCOL_ID
    value["world_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_WORLD_SEEDS
    value["sensor_seeds"]["dlolab_slingshot_v4"] = SLINGSHOT_SENSOR_SEEDS
    value["recovery"] = {
        "terminal_slingshot_v3_failure_id": TERMINAL_SLINGSHOT_V3_FAILURE_ID,
        "terminal_stage": "calibration-prefixes",
        "ordinary_prefix_seals": 0,
        "ordinary_future_seals": 0,
        "outcomes_opened": False,
        "worlds_reused": False,
        "scientific_change": False,
        "repair": "use_complete_tree_at_identical_frozen_upstream_commit",
        "scene_construction_preflight_passed": True,
        "scene_preflight_registered_worlds": 0,
        "scene_preflight_actions": 0,
    }
    value["complete_upstream"] = {
        "revision": "c5026a9416b03c6bc5186eba13cd4ffd4c0e7796",
        "working_tree_clean": True,
        "plane_asset_sha256": (
            "be1a566d558bd89cabfee5b65d13f3c76acd4c009e3eb8830b369b2dfa079d29"
        ),
    }
    value["protocol_id"] = content_id(value)
    return value


def score(outcomes: dict[str, QueryOutcomeV1]) -> dict[str, Any]:
    value = cast(dict[str, Any], score_v3(outcomes))
    value.pop("artifact_id")
    value["version"] = VERSION
    value["protocol_id"] = protocol()["protocol_id"]
    value["artifact_id"] = content_id(value)
    return value
