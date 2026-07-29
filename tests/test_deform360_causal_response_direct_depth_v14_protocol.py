from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    AdaptiveCausalResponseQueryConfig,
)
from bayesian_phystwin.deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_preflight import (
    AdaptiveDirectDepthSourcePreflightConfigV14,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_source_lock import (
    PROTOCOL_ID,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_synthetic import (
    validate_adaptive_direct_depth_synthetic_v14,
)
from bayesian_phystwin.deform360_causal_response_event import (
    CausalResponseEventConfig,
)
from bayesian_phystwin.deform360_causal_response_update import (
    CausalResponseMeasurementConfig,
)
from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
)
from bayesian_phystwin.deform360_object_exclusion import (
    load_object_exclusion_manifest,
)
from bayesian_phystwin.observation_belief import file_sha256
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sota" / "deform360_causal_response_direct_depth_v14.json"
EXCLUSION = ROOT / "configs" / "sota" / "deform360_fresh_object_exclusion_v14.json"
SYNTHETIC = (
    ROOT
    / "results"
    / "sota"
    / "deform360_causal_response_direct_depth_synthetic_v14"
    / "summary.json"
)


def _payload() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-v14-protocol\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_v14_protocol_binds_implementation_and_executable_defaults() -> None:
    payload = _payload()

    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["status"] == ("implementation-locked-before-fresh-source-selection")
    assert payload["config_sha256"] == _canonical_sha256(payload)
    assert payload["implementation_commit"] == (
        "2af99480f42e028c2612e5fe7d863b2183704040"
    )
    for relative, digest in payload["implementation_file_sha256"].items():
        assert file_sha256(ROOT / relative) == digest

    assert payload["source_preflight"] == json.loads(
        json.dumps(asdict(AdaptiveDirectDepthSourcePreflightConfigV14()))
    )
    assert payload["adaptive_carrier"] == asdict(AdaptiveCausalResponseQueryConfig())
    assert payload["event"] == asdict(CausalResponseEventConfig())
    strict = DirectDepthEndpointConfig()
    assert payload["direct_depth_arms"]["strict_3plus3"] == asdict(strict)
    assert payload["direct_depth_arms"]["inflated_2plus2"] == asdict(
        replace(
            strict,
            minimum_camera_support=2,
            correlation_covariance_inflation=4.0,
        )
    )
    assert payload["admission"] == asdict(CausalResponseAdmissionConfig())
    assert payload["measurement"] == asdict(CausalResponseMeasurementConfig())
    assert payload["belief"] == asdict(
        RecursiveRbfBeliefConfig(
            length_scale_fraction=0.10,
            local_blend=1.0,
        )
    )


def test_v14_protocol_binds_the_complete_hash_only_freshness_boundary() -> None:
    payload = _payload()
    freshness = payload["freshness_boundary"]
    exclusion = load_object_exclusion_manifest(EXCLUSION)

    assert freshness["manifest_owner"] == exclusion["owner"]
    assert freshness["manifest_exclusion_sha256"] == exclusion["exclusion_sha256"]
    assert freshness["manifest_file_sha256"] == file_sha256(EXCLUSION)
    assert freshness["excluded_physical_object_count"] == len(
        exclusion["object_hashes"]
    )
    assert len(exclusion["object_hashes"]) == 138
    assert freshness["prior_target_outcome_access_allowed"] is False


def test_v14_protocol_requires_transfer_safety_and_exact_fallback() -> None:
    payload = _payload()
    gate = payload["source_gate"]
    contract = payload["method_contract"]

    assert gate["required_prediction_or_exact_fallback_count"] == 12
    assert gate["maximum_technical_failure_count"] == 0
    assert gate["minimum_event_admitted_object_count"] == 6
    assert gate["minimum_object_balanced_hidden_identity_improvement_fraction"] == 0.05
    assert gate["minimum_object_balanced_chamfer_improvement_fraction"] == 0.05
    assert gate["minimum_joint_object_win_count"] == 8
    assert gate["maximum_single_object_regression_fraction"] == 0.05
    assert gate["maximum_false_safe_rate"] == 0.10
    assert gate["required_source_group_upper_regret_m"] < 0.0

    assert contract["fixed_identity_tracker_provider_reused"] is False
    assert contract["two_view_rows_require_fourfold_covariance_inflation"] is True
    assert contract["association_probability_separate_from_prior_reliability"]
    assert contract["state_innovation_changes_prior_reliability"] is False
    assert contract["innovation_robustified_once"] is True
    assert contract["any_rejection_is_bit_exact_selected_baseline"] is True


def test_v14_protocol_preserves_outcome_and_held_v8_boundaries() -> None:
    payload = _payload()
    boundary = payload["information_boundary"]
    controls = payload["synthetic_control_gate"]

    assert boundary["maximum_object_observation_frame"] == 57
    assert boundary["future_object_observation_allowed"] is False
    assert boundary["future_identity_allowed_before_all_prediction_seals"] is False
    assert boundary["source_outcomes_allowed_before_all_prediction_seals"] is False
    assert boundary["target_object_selection_or_outcome_allowed"] is False
    assert (
        boundary["held_v8_target_query_score_barrier_outcome_or_process_access_allowed"]
        is False
    )

    assert controls["trial_count_per_arm"] == 6
    assert controls["required_positive_detection_count"] == 12
    assert controls["maximum_placebo_admission_count"] == 0
    assert controls["required_exact_fallback_count"] == 12
    assert controls["must_pass_before_source_lock"] is True
    assert controls["real_data_evidence"] is False

    result = validate_adaptive_direct_depth_synthetic_v14(SYNTHETIC)
    registered = payload["synthetic_control_result"]
    assert registered["artifact_sha256"] == result.artifact_sha256
    assert registered["file_sha256"] == file_sha256(SYNTHETIC)
    assert registered["positive_detection_count"] == 12
    assert registered["placebo_admission_count"] == 0
    assert registered["placebo_exact_fallback_count"] == 12
    assert registered["gate_passed"] is True
    assert registered["real_data_evidence"] is False
