from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.deform360_direct_depth_action_response import (
    DirectDepthActionResponseConfig,
)
from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
    DYNAMIC_DEPTH_ENDPOINT_PAIRS,
    Deform360SentinelQueryConfig,
)


def test_v9_source_protocol_matches_typed_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "configs/sota/deform360_dynamic_direct_depth_admission_source_v9.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["protocol_id"] == DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID
    assert tuple(map(tuple, payload["endpoint_pairs"])) == DYNAMIC_DEPTH_ENDPOINT_PAIRS
    assert DirectDepthActionResponseConfig(**payload["admission"]) == (
        DirectDepthActionResponseConfig()
    )
    depth = {
        key: value
        for key, value in payload["direct_depth"].items()
        if key != "unknown_cross_view_correlation"
    }
    assert DirectDepthEndpointConfig(**depth) == DirectDepthEndpointConfig()
    for birth_frame, update_frame in DYNAMIC_DEPTH_ENDPOINT_PAIRS:
        Deform360SentinelQueryConfig(
            **payload["query_schedule"],
            query_birth_frame=birth_frame,
            query_update_frame=update_frame,
            protocol_id=DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
        )


def test_v9_source_panel_is_unique_and_outcome_blind() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "configs/sota/deform360_dynamic_direct_depth_admission_source_v9.json"
        ).read_text(encoding="utf-8")
    )
    cases = payload["cases"]

    assert len(cases) == payload["source_gate"]["locked_case_count"] == 7
    assert len({record["case"] for record in cases}) == 7
    assert all(len(record["physical_archive_sha256"]) == 64 for record in cases)
    assert payload["source_gate"]["minimum_admitted_case_count"] == 4
    assert payload["source_gate"]["candidate_state_updates_allowed_during_gate"] is False
    assert payload["information_boundary"]["future_metric_read"] is False
    assert payload["information_boundary"]["held_v8_read"] is False
