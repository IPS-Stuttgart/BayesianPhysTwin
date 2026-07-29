from __future__ import annotations

import json
from pathlib import Path

from bayesian_phystwin.observation_belief import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results"
    / "sota"
    / "diagnostics"
    / "deform360_causal_response_tracker_v13_source"
)


def test_v13_tracker_source_result_is_frozen_negative_evidence() -> None:
    barrier = RESULT_ROOT / "prediction_completeness_barrier.json"
    result_path = RESULT_ROOT / "source_competence_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert file_sha256(barrier) == (
        "d796a315568de16ec0af938903f9d50aa17f016731f356d87014ec6e6eb1cba3"
    )
    assert file_sha256(result_path) == (
        "9ca4e1f6b4c9d5d5e12d019e3ad9fe29cec6211b57799ee54b918095f719abe1"
    )
    assert result["result_sha256"] == (
        "cc8752943761a03c0992a47f65d150102e5b896cbc26284399a13c316b26c744"
    )
    assert result["source_gate_passed"] is False
    assert result["decision"] == "stop_v13_tracker_provider_route"

    aggregate = result["aggregate"]
    assert aggregate["provider_prediction_count"] == 6
    assert aggregate["exact_query_abstention_count"] == 2
    assert aggregate["supported_identity_count"] == 5
    assert aggregate["scheduled_identity_count"] == 96
    assert aggregate["pooled_supported_fraction"] == 5 / 96
    assert aggregate["scored_case_count"] == 1
    assert aggregate["provider_case_wins"] == 1
    assert aggregate["relative_gain_over_persistence"] > 0.17
    assert aggregate["mean_accepted_cross_panel_disagreement_m"] < 0.003

    gates = result["gates"]
    assert gates["provider_prediction_count"] is True
    assert gates["relative_gain_over_persistence"] is True
    assert gates["cross_panel_disagreement"] is True
    assert gates["pooled_supported_fraction"] is False
    assert gates["case_supported_fraction"] is False
    assert gates["scored_case_count"] is False
    assert gates["provider_case_wins"] is False
    assert gates["object_balanced_rmse"] is False
    assert gates["object_balanced_late_rmse"] is False

    boundary = result["information_boundary"]
    assert boundary["maximum_scored_frame"] == 57
    assert boundary["state_or_readout_update_constructed"] is False
    assert boundary["future_prediction_metric_read"] is False
    assert boundary["held_v8_artifact_or_process_access"] is False
