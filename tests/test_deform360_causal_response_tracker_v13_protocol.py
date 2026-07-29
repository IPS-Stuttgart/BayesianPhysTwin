from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_tracker import PROTOCOL_ID
from bayesian_phystwin.observation_belief import file_sha256
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY,
    DynamicMultiviewConfig,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    DynamicTAPNextPPRuntimeConfig,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "sota"
    / "deform360_causal_response_tracker_v13.json"
)


def _payload() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-tracker-v13-protocol\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_protocol_is_hash_locked_to_implementation_and_v13_carriers() -> None:
    payload = _payload()

    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["status"] == "locked-before-source-tracker-outcomes"
    assert payload["config_sha256"] == _canonical_sha256(payload)
    assert payload["implementation_commit"] == (
        "d4f43b24117b68ebd611fd97cc1d1feb68b30d14"
    )
    assert payload["parent_v13_source_result_sha256"] == (
        "63a9ffde07de4e0378605b8574a8d4a8acfe6a382c65df2be185c47f6276239c"
    )
    for relative, digest in payload["implementation_file_sha256"].items():
        assert file_sha256(ROOT / relative) == digest

    cases = payload["cases"]
    assert len(cases) == len({row["case"] for row in cases}) == 8
    assert sum(row["arm"] == "strict_3plus3" for row in cases) == 2
    assert sum(row["arm"] == "inflated_2plus2" for row in cases) == 4
    assert (
        sum(row["arm"] == "abstained_insufficient_2plus2" for row in cases)
        == 2
    )
    assert all(len(row["adaptive_query_result_sha256"]) == 64 for row in cases)
    assert all(len(row["adaptive_query_report_sha256"]) == 64 for row in cases)


def test_protocol_preserves_correlation_and_information_boundaries() -> None:
    payload = _payload()
    tracker = DynamicTAPNextPPRuntimeConfig(**payload["tracker_runtime"])
    multiview = payload["multiview"]

    assert tracker.input_resolution == 512
    assert multiview["assignment_uncertainty_mode"] == (
        COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY
    )
    strict = DynamicMultiviewConfig(
        **{
            **multiview,
            "minimum_claim_view_count": payload["arm_settings"][
                "strict_3plus3"
            ]["minimum_claim_view_count"],
        }
    )
    fallback = DynamicMultiviewConfig(
        **{
            **multiview,
            "minimum_claim_view_count": payload["arm_settings"][
                "inflated_2plus2"
            ]["minimum_claim_view_count"],
        }
    )
    assert strict.minimum_claim_view_count == 3
    assert fallback.minimum_claim_view_count == 2
    assert fallback.two_view_covariance_inflation == 4.0
    assert fallback.shared_bias_standard_deviation_m == 0.005

    boundary = payload["information_boundary"]
    assert boundary["proposal_and_validation_camera_panels_disjoint"] is True
    assert boundary["physical_innovation_available_to_prior_reliability"] is False
    assert boundary["identity_target_available_before_prediction_barrier"] is False
    assert boundary["state_or_readout_update_allowed"] is False
    assert boundary["future_prediction_metric_allowed"] is False
    assert boundary["held_v8_access_allowed"] is False


def test_source_gate_requires_transfer_before_update_development() -> None:
    gate = _payload()["source_gate"]

    assert gate["locked_case_count"] == 8
    assert gate["required_provider_prediction_count"] == 6
    assert gate["minimum_pooled_supported_fraction"] == 0.5
    assert gate["minimum_case_support_pass_count"] == 5
    assert gate["minimum_scored_case_count"] == 5
    assert gate["maximum_object_balanced_rmse_m"] == 0.015
    assert gate["maximum_object_balanced_late_rmse_m"] == 0.015
    assert gate["minimum_relative_gain_over_persistence"] == 0.1
    assert gate["minimum_provider_case_wins"] == 4
    assert gate["maximum_mean_cross_panel_disagreement_m"] == 0.01
