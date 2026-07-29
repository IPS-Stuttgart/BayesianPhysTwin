from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_query import (
    CausalResponseQueryConfig,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "sota" / "deform360_causal_response_query_feasibility_v12.json"
)


def _canonical_sha256(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_query_feasibility_protocol_binds_the_frozen_v12_method() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["config_sha256"] == _canonical_sha256(payload)
    assert payload["parent_method_commit"] == (
        "d5eab1b1dcf8bb77cd7a37f9716f5846559e930c"
    )
    assert payload["query"] == asdict(CausalResponseQueryConfig())
    assert len(payload["cases"]) == 8
    assert len({row["case"] for row in payload["cases"]}) == 8
    assert payload["source_feasibility_gate"] == {
        "locked_case_count": 8,
        "minimum_admitted_case_count": 6,
        "maximum_technical_failure_count": 0,
        "pass_action": (
            "Stage released tactile streams and run the separately frozen V12 "
            "prefix event/admission path without changing the query method."
        ),
        "fail_action": (
            "Stop V12 without weakening query count, panel support, association, "
            "action support, or source gate settings."
        ),
    }


def test_query_feasibility_protocol_stops_before_outcome_or_update() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    boundary = payload["information_boundary"]

    assert boundary["maximum_object_observation_frame"] == 0
    assert boundary["source_cases_previously_opened_by_v10_v11"] is True
    assert boundary["tactile_read"] is False
    assert boundary["tracker_read"] is False
    assert boundary["future_identity_read"] is False
    assert boundary["future_object_observation_read"] is False
    assert boundary["future_metric_read"] is False
    assert boundary["state_update_constructed"] is False
    assert boundary["v1_sealed_target_allowed"] is False
    assert boundary["held_v8_artifact_or_process_access_allowed"] is False
