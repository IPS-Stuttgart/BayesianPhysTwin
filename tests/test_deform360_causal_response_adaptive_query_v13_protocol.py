from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    AdaptiveCausalResponseQueryConfig,
)
from bayesian_phystwin.observation_belief import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sota" / "deform360_causal_response_adaptive_query_v13.json"
V12_CONFIG = (
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


def test_adaptive_query_protocol_binds_implementation_and_source_panel() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    v12 = json.loads(V12_CONFIG.read_text(encoding="utf-8"))

    assert payload["config_sha256"] == _canonical_sha256(payload)
    assert payload["implementation_commit"] == (
        "bdde57f790fdc6c41255d6968e1387e97c381062"
    )
    assert payload["parent_v12_result_commit"] == (
        "cf2b532e01a3c92d1761f5bdea36a7c026e2c3b8"
    )
    assert payload["query"] == asdict(AdaptiveCausalResponseQueryConfig())
    assert payload["cases"] == v12["cases"]
    assert len(payload["cases"]) == 8
    for relative, digest in payload["implementation_file_sha256"].items():
        assert file_sha256(ROOT / relative) == digest


def test_adaptive_query_protocol_has_a_strict_source_gate() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["source_feasibility_gate"] == {
        "locked_case_count": 8,
        "minimum_admitted_case_count": 6,
        "minimum_strict_admitted_case_count": 2,
        "maximum_technical_failure_count": 0,
        "pass_action": (
            "Freeze a separate source-only tactile-event, tracker-competence, "
            "bias-aware update, and exact-fallback study without changing the "
            "V13 carrier."
        ),
        "fail_action": (
            "Stop V13 without changing the panel objective, support ladder, "
            "covariance inflation, shared-bias scale, query budget, or source gate."
        ),
    }


def test_adaptive_query_protocol_stops_before_outcome_or_update() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    boundary = payload["information_boundary"]

    assert boundary["maximum_object_observation_frame"] == 0
    assert boundary["source_cases_previously_opened_by_v10_v11_v12"] is True
    assert boundary["known_physical_action_support_used"] is True
    assert boundary["complete_camera_panel_selected_target_free"] is True
    assert boundary["tactile_read"] is False
    assert boundary["tracker_read"] is False
    assert boundary["future_identity_read"] is False
    assert boundary["future_object_observation_read"] is False
    assert boundary["future_metric_read"] is False
    assert boundary["state_update_constructed"] is False
    assert boundary["v1_sealed_target_allowed"] is False
    assert boundary["held_v8_artifact_or_process_access_allowed"] is False
