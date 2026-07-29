from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bayesian_phystwin.deform360_causal_response_admission import (
    CausalResponseAdmissionConfig,
)
from bayesian_phystwin.deform360_causal_response_event import (
    CausalResponseEventConfig,
)
from bayesian_phystwin.deform360_causal_response_prefix import (
    CausalResponsePrefixConfig,
)
from bayesian_phystwin.deform360_causal_response_query import (
    CausalResponseQueryConfig,
)
from bayesian_phystwin.deform360_causal_response_update import (
    CausalResponseMeasurementConfig,
)
from bayesian_phystwin.phystwin_online_belief import RecursiveRbfBeliefConfig

CONFIG = Path("configs/sota/deform360_causal_response_method_v12.json")


def _canonical_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_v12_method_lock_matches_the_executable_defaults() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["protocol_id"] == "deform360-causal-response-depth-v12"
    assert (
        payload["status"]
        == "method-locked-awaiting-complete-hash-only-cohort-exclusion"
    )
    assert payload["config_sha256"] == _canonical_sha256(payload)
    assert payload["prefix"] == asdict(CausalResponsePrefixConfig())
    assert payload["event"] == asdict(CausalResponseEventConfig())
    assert payload["query"] == asdict(CausalResponseQueryConfig())
    assert payload["admission"] == asdict(CausalResponseAdmissionConfig())
    assert payload["measurement"] == asdict(CausalResponseMeasurementConfig())
    assert payload["belief"] == asdict(
        RecursiveRbfBeliefConfig(
            length_scale_fraction=0.10,
            local_blend=1.0,
        )
    )


def test_v12_cannot_select_a_cohort_without_the_held_exclusion_manifest() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    freshness = payload["freshness_boundary"]

    assert freshness["cohort_selection_allowed"] is False
    assert freshness["missing_at_method_lock"] == [
        "independently supplied held-v8 all-attempt hash-only exclusion manifest"
    ]
    assert (
        freshness["object_ids_or_outcomes_may_be_read_to_complete_exclusions"] is False
    )
    assert "cases" not in payload


def test_v12_preserves_the_information_and_exact_fallback_boundaries() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    boundary = payload["information_boundary"]
    contract = payload["method_contract"]
    panels = payload["camera_panels"]

    assert boundary["future_object_observation_allowed"] is False
    assert boundary["future_identity_allowed_before_prediction_seal"] is False
    assert (
        boundary["held_v8_target_query_score_barrier_outcome_or_process_access_allowed"]
        is False
    )
    assert contract["association_probability_separate_from_prior_reliability"] is True
    assert contract["rejection_is_bit_exact_selected_baseline"] is True
    assert contract["candidate_is_readout_discrepancy_not_warp_state_injection"] is True
    assert set(panels["proposal_panel_indices"]).isdisjoint(
        panels["validation_panel_indices"]
    )
    assert sorted(
        panels["proposal_panel_indices"] + panels["validation_panel_indices"]
    ) == list(range(len(panels["full_panel"])))
